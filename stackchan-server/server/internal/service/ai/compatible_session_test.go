package ai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCompatibleSilenceCompletesProtocolWithoutSpeech(t *testing.T) {
	ttsRequests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/audio/transcriptions":
			_, _ = w.Write([]byte(`{"text":"","segments":[]}`))
		case "/v1/audio/speech":
			ttsRequests++
			_, _ = w.Write([]byte{0x34, 0x12})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	idleCalls := 0
	noSpeechCalls := 0
	startCalls := 0
	stopCalls := 0
	audioSamples := 0
	session := &compatibleSession{
		sttClient: newOpenAIClient(server.URL, "test-key", "", "whisper-1", "ko", "", "", "", ""),
		ttsClient: newOpenAIClient(server.URL, "test-key", "", "", "", "gpt-4o-mini-tts", "coral", "", ""),
		cb: RealtimeCallbacks{
			OnIdle:     func() { idleCalls++ },
			OnNoSpeech: func() { noSpeechCalls++ },
			OnStart:    func() { startCalls++ },
			OnAudio:    func(pcm []int16) { audioSamples += len(pcm) },
			OnStop:     func() { stopCalls++ },
		},
	}
	session.completeTurn(context.Background(), make([]int16, 160))
	if ttsRequests != 0 {
		t.Fatalf("silence unexpectedly requested TTS %d times", ttsRequests)
	}
	if idleCalls != 0 || noSpeechCalls != 1 || startCalls != 1 || stopCalls != 1 || audioSamples != 0 {
		t.Fatalf("idle=%d no_speech=%d start=%d stop=%d audio=%d", idleCalls, noSpeechCalls, startCalls, stopCalls, audioSamples)
	}
}
