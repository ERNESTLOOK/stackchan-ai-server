package ai

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gogf/gf/v2/frame/g"
	"github.com/gogf/gf/v2/os/gcfg"
	"github.com/gorilla/websocket"
)

func TestServerNeverInitiatesDeviceListening(t *testing.T) {
	source, err := os.ReadFile("ws_simulator.go")
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(source, []byte(`"type": "listen"`)) {
		t.Fatal("server must never send a listen command; the current firmware disconnects after receiving one")
	}
}

func TestConversationSilenceDoesNotExtendFollowUpWindow(t *testing.T) {
	now := time.Now()
	a := newConversationActivity(15*time.Second, now)
	for seconds := 1; seconds < 15; seconds++ {
		if !a.audio(now.Add(time.Duration(seconds)*time.Second), []int16{0, 1, -1}) {
			t.Fatal("window expired too soon")
		}
	}
	if a.commit(now.Add(14 * time.Second)) {
		t.Fatal("silence-only listen:stop extended the window")
	}
	if a.audio(now.Add(15*time.Second), []int16{3000}) {
		t.Fatal("speech after expiry reopened the microphone")
	}
	if a.responding(now.Add(16 * time.Second)) {
		t.Fatal("late provider response reopened expired conversation")
	}
}

func TestConversationAllowsFollowUpAndWaitsForPlayback(t *testing.T) {
	now := time.Now()
	a := newConversationActivity(15*time.Second, now)
	if !a.audio(now.Add(14*time.Second), []int16{1000, -1000}) {
		t.Fatal("follow-up rejected")
	}
	if a.expired(now.Add(20 * time.Second)) {
		t.Fatal("active utterance was cut off")
	}
	if !a.commit(now.Add(21 * time.Second)) {
		t.Fatal("speech was not committed")
	}
	if !a.responding(now.Add(30 * time.Second)) {
		t.Fatal("provider response rejected")
	}
	if a.expired(now.Add(50 * time.Second)) {
		t.Fatal("expired while assistant was speaking")
	}
	a.playbackDone(now.Add(60 * time.Second))
	if a.expired(now.Add(74*time.Second)) || !a.expired(now.Add(75*time.Second)) {
		t.Fatal("follow-up window did not start at playback completion")
	}
}

func TestAutonomousActionWaitsForProviderResponse(t *testing.T) {
	now := time.Now()
	a := newConversationActivity(15*time.Second, now)
	if !a.audio(now, []int16{1000, -1000}) || !a.commit(now) {
		t.Fatal("could not enter provider response state")
	}
	s := &wsSession{activity: a, frameQueue: make(chan []byte, 1)}
	if s.canStartAutonomousAction() {
		t.Fatal("autonomous action started while STT/LLM/TTS response was pending")
	}
	a.playbackDone(now)
	if !s.canStartAutonomousAction() {
		t.Fatal("autonomous action did not resume after response completed")
	}
	s.finishAutonomousAction()
}

func TestConversationAutoListenDoesNotRearmTimeout(t *testing.T) {
	now := time.Now()
	a := newConversationActivity(15*time.Second, now)
	s := &wsSession{activity: a}
	s.handleListen(context.Background(), map[string]any{"state": "start", "mode": "auto"})
	if !a.deadline.Equal(now.Add(15 * time.Second)) {
		t.Fatal("automatic listen:start reset the deadline")
	}
}

func TestConversationWakeAndDisabledTimeout(t *testing.T) {
	now := time.Now()
	a := newConversationActivity(15*time.Second, now)
	a.wake(now.Add(10 * time.Second))
	if a.expired(now.Add(20 * time.Second)) {
		t.Fatal("wake word did not refresh window")
	}
	disabled := newConversationActivity(0, now)
	if disabled.expired(now.Add(24*time.Hour)) || !disabled.audio(now.Add(25*time.Hour), nil) {
		t.Fatal("legacy unlimited mode expired")
	}
	waiting := newConversationActivity(15*time.Second, now)
	waiting.responding(now)
	if !waiting.expired(now.Add(2 * time.Minute)) {
		t.Fatal("stalled provider held the connection indefinitely")
	}
}

func TestConversationRuntimeDefaults(t *testing.T) {
	t.Setenv("STACKCHAN_DATA_DIR", t.TempDir())
	previous := g.Cfg().GetAdapter()
	t.Cleanup(func() { g.Cfg().SetAdapter(previous) })
	for _, test := range []struct {
		config string
		want   int
	}{
		{`{"ai":{"ha_enabled":false}}`, 15},
		{`{"ai":{"ha_enabled":false,"standalone_ha_enabled":true}}`, 15},
		{`{"ai":{"ha_enabled":true}}`, 0},
	} {
		adapter, err := gcfg.NewAdapterContent(test.config)
		if err != nil {
			t.Fatal(err)
		}
		g.Cfg().SetAdapter(adapter)
		if got := conversationIdleSeconds(context.Background()); got != test.want {
			t.Fatalf("timeout=%d, want %d", got, test.want)
		}
	}
}

type idleTestProvider struct{}

func (idleTestProvider) AppendAudio([]int16) error { return nil }
func (idleTestProvider) CommitAudio() error        { return nil }
func (idleTestProvider) CancelResponse() error     { return nil }
func (idleTestProvider) Close()                    {}

type commitCountingProvider struct{ commits int }

func (p *commitCountingProvider) AppendAudio([]int16) error { return nil }
func (p *commitCountingProvider) CommitAudio() error        { p.commits++; return nil }
func (p *commitCountingProvider) CancelResponse() error     { return nil }
func (p *commitCountingProvider) Close()                    {}

func TestCompatibleServerVADCommitsAfterTrailingSilence(t *testing.T) {
	p := &commitCountingProvider{}
	s := &wsSession{
		rt:          p,
		activity:    newConversationActivity(0, time.Now()),
		isListening: true,
		serverVAD:   true,
	}
	s.observeServerVAD(context.Background(), []int16{2000, -2000})
	silence := make([]int16, serverSampleRate/20)
	for i := 0; i < 7; i++ {
		committed := s.observeServerVAD(context.Background(), silence)
		if i < 7 && committed {
			t.Fatal("committed before configured trailing silence")
		}
	}
	if !s.observeServerVAD(context.Background(), silence) {
		t.Fatal("did not commit after configured trailing silence")
	}
	if p.commits != 1 || s.isListening {
		t.Fatalf("commits=%d listening=%t, want one commit and stopped listening", p.commits, s.isListening)
	}
}

func TestApplyPCMVolumeClampsThreeTimesGain(t *testing.T) {
	pcm := []int16{1000, -1000, 20000, -20000, 0}
	applyPCMVolume(pcm, 3)
	want := []int16{3000, -3000, 32767, -32768, 0}
	for i := range want {
		if pcm[i] != want[i] {
			t.Fatalf("pcm[%d]=%d, want %d", i, pcm[i], want[i])
		}
	}
}

func TestApplyPCMPitchRateSpeedsUpAndClamps(t *testing.T) {
	pcm := []int16{0, 1000, 2000, 3000, 4000, 5000}
	got := applyPCMPitchRate(pcm, 1.2)
	if len(got) != 5 {
		t.Fatalf("len=%d, want 5", len(got))
	}
	if got[0] != 0 || got[1] != 1200 || got[4] != 4800 {
		t.Fatalf("pitch output=%v", got)
	}
	if len(applyPCMPitchRate(pcm, 2.0)) != 4 {
		t.Fatal("pitch rate was not clamped to the supported upper bound")
	}
}

func TestLoudSoundDetectedOnlyForStartleLevelAudio(t *testing.T) {
	normalSpeech := []int16{600, -900, 1200, -1400, 900, -700}
	if loudSoundDetected(normalSpeech) {
		t.Fatal("normal speech level triggered loud startle")
	}
	sharpClap := []int16{400, -800, 23000, -1200, 700}
	if !loudSoundDetected(sharpClap) {
		t.Fatal("sharp loud peak did not trigger startle")
	}
	loudSustained := []int16{9500, -9500, 9200, -9200}
	if !loudSoundDetected(loudSustained) {
		t.Fatal("sustained loud audio did not trigger startle")
	}
}

func TestConversationIdleClosesDeviceAudioChannel(t *testing.T) {
	done := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(done)
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		s := &wsSession{conn: conn, rt: idleTestProvider{}, activity: newConversationActivity(200*time.Millisecond, time.Now()), sessionID: "test-session", frameQueue: make(chan []byte, 2)}
		go s.idleLoop(ctx)
		s.run(ctx)
	}))
	defer server.Close()
	conn, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if err := conn.WriteJSON(map[string]any{"type": "hello"}); err != nil {
		t.Fatal(err)
	}
	if err := conn.SetReadDeadline(time.Now().Add(3 * time.Second)); err != nil {
		t.Fatal(err)
	}
	var hello map[string]any
	if err := conn.ReadJSON(&hello); err != nil || hello["type"] != "hello" {
		t.Fatalf("hello: %v %v", hello, err)
	}
	if err := conn.WriteJSON(map[string]any{"type": "listen", "state": "start", "mode": "auto"}); err != nil {
		t.Fatal(err)
	}
	_, _, err = conn.ReadMessage()
	if !websocket.IsCloseError(err, websocket.CloseNormalClosure) {
		t.Fatalf("wanted normal idle close, got %v", err)
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("device handler did not stop after idle close")
	}
}
