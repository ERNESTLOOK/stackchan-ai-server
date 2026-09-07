/*
SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
SPDX-License-Identifier: MIT
*/

package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestExplainImageUsesMultimodalChat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path=%s", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		messages := body["messages"].([]any)
		content := messages[0].(map[string]any)["content"].([]any)
		imageURL := content[1].(map[string]any)["image_url"].(map[string]any)["url"].(string)
		if !strings.HasPrefix(imageURL, "data:image/jpeg;base64,") {
			t.Fatalf("image URL=%q", imageURL)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]any{"content": "책상이 보여."}}}})
	}))
	defer server.Close()

	client := newOpenAIClient(server.URL+"/v1", "key", "vision-model", "", "", "", "", "")
	got, err := client.ExplainImage(context.Background(), []byte{0xff, 0xd8, 0xff}, "뭐가 보여?")
	if err != nil || got != "책상이 보여." {
		t.Fatalf("got=%q err=%v", got, err)
	}
}

func TestDetectImageMediaTypeAllowsPNG(t *testing.T) {
	png := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d}
	if got := detectImageMediaType(png); got != "image/png" {
		t.Fatalf("media type=%q, want image/png", got)
	}
}
