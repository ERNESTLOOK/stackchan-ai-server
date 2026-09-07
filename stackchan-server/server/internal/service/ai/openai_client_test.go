package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
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

	client := newOpenAIClient(server.URL+"/v1", "test-key", "test-model", "", "", "", "", "", "")
	reply, err := client.Chat(context.Background(), []chatMessage{{Role: "user", Content: "hello"}}, nil, nil)
	if err != nil || reply != "ok" {
		t.Fatalf("reply=%q err=%v", reply, err)
	}
}

func TestCompatibleChatRetriesPromptLimitWithCurrentTurnOnly(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var body struct {
			Messages []chatMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if requests == 1 {
			w.WriteHeader(http.StatusPaymentRequired)
			_, _ = w.Write([]byte(`{"error":{"message":"Prompt tokens limit exceeded: 4000 > 2500"}}`))
			return
		}
		if len(body.Messages) != 2 || body.Messages[0].Role != "system" || body.Messages[1].Content != "최신 질문" {
			t.Fatalf("retry messages=%#v", body.Messages)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"됐어, 교수님!"}}]}`))
	}))
	defer server.Close()

	client := newOpenAIClient(server.URL, "test-key", "test-model", "", "", "", "", "", "persona")
	reply, err := client.Chat(context.Background(), []chatMessage{
		{Role: "user", Content: "오래된 질문"},
		{Role: "assistant", Content: "오래된 답변"},
		{Role: "user", Content: "최신 질문"},
	}, nil, nil)
	if err != nil || reply != "됐어, 교수님!" || requests != 2 {
		t.Fatalf("reply=%q requests=%d err=%v", reply, requests, err)
	}
}

func TestPromptLimitTrimPreservesTwoRecentHistoryPairs(t *testing.T) {
	messages := []chatMessage{{Role: "system", Content: "persona"}}
	for index := 0; index < 4; index++ {
		messages = append(messages,
			chatMessage{Role: "user", Content: fmt.Sprintf("질문-%d", index)},
			chatMessage{Role: "assistant", Content: fmt.Sprintf("답변-%d", index)},
		)
	}
	messages = append(messages, chatMessage{Role: "user", Content: "현재 질문"})
	trimmed, dropped := trimOlderConversationContext(messages, fmt.Errorf("Prompt tokens limit exceeded"))
	if dropped != 4 || len(trimmed) != 6 {
		t.Fatalf("dropped=%d len=%d messages=%#v", dropped, len(trimmed), trimmed)
	}
	if trimmed[1].Content != "질문-2" || trimmed[len(trimmed)-1].Content != "현재 질문" {
		t.Fatalf("wrong history preserved: %#v", trimmed)
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

	client := newOpenAIClient(server.URL+"/v1", "test-key", "", "", "", "gpt-4o-mini-tts", "nova", "bright and playful", "")
	pcm, err := client.Speak(context.Background(), "hello")
	if err != nil || len(pcm) != 1 || pcm[0] != 0x1234 {
		t.Fatalf("pcm=%v err=%v", pcm, err)
	}
}

func TestWhisperTranscriptionUsesKoreanConfidenceMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		for key, want := range map[string]string{
			"model": "whisper-1", "language": "ko", "temperature": "0", "response_format": "verbose_json",
		} {
			if got := r.FormValue(key); got != want {
				t.Fatalf("%s=%q, want %q", key, got, want)
			}
		}
		_, _ = w.Write([]byte(`{"text":"안녕하세요","segments":[{"no_speech_prob":0.02,"avg_logprob":-0.2}]}`))
	}))
	defer server.Close()

	client := newOpenAIClient(server.URL+"/v1", "test-key", "", "whisper-1", "ko", "", "", "", "")
	text, err := client.Transcribe(context.Background(), []byte("wav"))
	if err != nil || text != "안녕하세요" {
		t.Fatalf("text=%q err=%v", text, err)
	}
}

func TestNonWhisperTranscriptionKeepsCompatibleJSONFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if got := r.FormValue("response_format"); got != "" {
			t.Fatalf("response_format=%q, want provider default", got)
		}
		_, _ = w.Write([]byte(`{"text":"테스트"}`))
	}))
	defer server.Close()

	client := newOpenAIClient(server.URL, "test-key", "", "gpt-4o-mini-transcribe", "ko", "", "", "", "")
	text, err := client.Transcribe(context.Background(), []byte("wav"))
	if err != nil || text != "테스트" {
		t.Fatalf("text=%q err=%v", text, err)
	}
}

func TestSTTResponseIsSilence(t *testing.T) {
	tests := []struct {
		name string
		resp sttResponse
		want bool
	}{
		{"empty text", sttResponse{}, true},
		{"plain compatible JSON", sttResponse{Text: "안녕"}, false},
		{"high no-speech hallucination", sttResponse{Text: "시청해주셔서 감사합니다", Segments: []sttSegment{{NoSpeechProb: 0.92, AvgLogprob: -0.4}}}, true},
		{"low-confidence garble", sttResponse{Text: "...", Segments: []sttSegment{{NoSpeechProb: 0.2, AvgLogprob: -1.8}}}, true},
		{"real speech", sttResponse{Text: "교수님 안녕하세요", Segments: []sttSegment{{NoSpeechProb: 0.03, AvgLogprob: -0.25}}}, false},
		{"one real segment", sttResponse{Text: "네 좋아요", Segments: []sttSegment{{NoSpeechProb: 0.95, AvgLogprob: -1.4}, {NoSpeechProb: 0.05, AvgLogprob: -0.4}}}, false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.resp.isSilence(); got != test.want {
				t.Fatalf("isSilence()=%t, want %t for %s", got, test.want, strings.TrimSpace(test.resp.Text))
			}
		})
	}
}
