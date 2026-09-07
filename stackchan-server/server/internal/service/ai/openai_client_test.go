package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCompatibleEndpointAvoidsDuplicateV1(t *testing.T) {
	tests := []struct {
		base, path, want string
	}{
		{"https://api.openai.com", "/v1/audio/transcriptions", "https://api.openai.com/v1/audio/transcriptions"},
		{"https://api.openai.com/v1", "/v1/audio/transcriptions", "https://api.openai.com/v1/audio/transcriptions"},
		{"https://openrouter.ai/api/v1/", "/v1/chat/completions", "https://openrouter.ai/api/v1/chat/completions"},
	}
	for _, test := range tests {
		if got := compatibleEndpoint(test.base, test.path); got != test.want {
			t.Fatalf("compatibleEndpoint(%q, %q) = %q, want %q", test.base, test.path, got, test.want)
		}
	}
}

func TestCompatibleChatLimitsOutputTokens(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["max_tokens"] != float64(compatibleChatMaxTokens) {
			t.Fatalf("max_tokens=%v, want %d", body["max_tokens"], compatibleChatMaxTokens)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer server.Close()

	client := newOpenAIClient(server.URL+"/v1", "test-key", "test-model", "", "", "", "", "")
	reply, err := client.Chat(context.Background(), []chatMessage{{Role: "user", Content: "hello"}}, nil, nil)
	if err != nil || reply != "ok" {
		t.Fatalf("reply=%q err=%v", reply, err)
	}
}

func TestSpeechRequestIncludesVoiceInstructions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["model"] != "gpt-4o-mini-tts" || body["voice"] != "nova" || body["instructions"] != "bright and playful" {
			t.Fatalf("unexpected speech request: %#v", body)
		}
		_, _ = w.Write([]byte{0x34, 0x12})
	}))
	defer server.Close()

	client := newOpenAIClient(server.URL+"/v1", "test-key", "", "", "gpt-4o-mini-tts", "nova", "bright and playful", "")
	pcm, err := client.Speak(context.Background(), "hello")
	if err != nil || len(pcm) != 1 || pcm[0] != 0x1234 {
		t.Fatalf("pcm=%v err=%v", pcm, err)
	}
}
