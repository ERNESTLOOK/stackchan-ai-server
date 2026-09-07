package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCompatibleSilenceSpeaksRetryAndCompletesPlayback(t *testing.T) {
	spokenText := ""
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/audio/transcriptions":
			_, _ = w.Write([]byte(`{"text":"","segments":[]}`))
		case "/v1/audio/speech":
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			spokenText, _ = body["input"].(string)
			_, _ = w.Write([]byte{0x34, 0x12})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	idleCalls := 0
	startCalls := 0
	stopCalls := 0
	audioSamples := 0
	session := &compatibleSession{
		sttClient: newOpenAIClient(server.URL, "test-key", "", "whisper-1", "ko", "", "", "", ""),
		ttsClient: newOpenAIClient(server.URL, "test-key", "", "", "", "gpt-4o-mini-tts", "coral", "", ""),
		cb: RealtimeCallbacks{
			OnIdle:  func() { idleCalls++ },
			OnStart: func() { startCalls++ },
			OnAudio: func(pcm []int16) { audioSamples += len(pcm) },
			OnStop:  func() { stopCalls++ },
		},
	}
	session.completeTurn(context.Background(), make([]int16, 160))
	if spokenText != "응? 교수님, 다시 한 번 말해 줘." {
		t.Fatalf("retry prompt=%q", spokenText)
	}
	if idleCalls != 0 || startCalls != 1 || stopCalls != 1 || audioSamples != 1 {
		t.Fatalf("idle=%d start=%d stop=%d audio=%d", idleCalls, startCalls, stopCalls, audioSamples)
	}
}
