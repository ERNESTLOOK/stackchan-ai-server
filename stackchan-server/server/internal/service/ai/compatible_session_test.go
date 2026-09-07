package ai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCompatibleSilenceReleasesResponseBusyState(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"text":"","segments":[]}`))
	}))
	defer server.Close()

	idleCalls := 0
	session := &compatibleSession{
		sttClient: newOpenAIClient(server.URL, "test-key", "", "whisper-1", "ko", "", "", "", ""),
		cb: RealtimeCallbacks{
			OnIdle: func() { idleCalls++ },
		},
	}
	session.completeTurn(context.Background(), make([]int16, 160))
	if idleCalls != 1 {
		t.Fatalf("OnIdle calls=%d, want 1", idleCalls)
	}
}
