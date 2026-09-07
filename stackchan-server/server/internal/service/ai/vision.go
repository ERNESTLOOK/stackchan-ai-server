/*
SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
SPDX-License-Identifier: MIT
*/

package ai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const maxVisionImageBytes = 8 << 20

func HandleVision(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		_ = json.NewEncoder(w).Encode(map[string]any{"success": false, "message": "POST required"})
		return
	}
	ctx := r.Context()
	configuredToken := aiString(ctx, "vision_token", "")
	if configuredToken != "" && r.Header.Get("Authorization") != "Bearer "+configuredToken {
		w.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(w).Encode(map[string]any{"success": false, "message": "unauthorized"})
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxVisionImageBytes+(1<<20))
	if err := r.ParseMultipartForm(maxVisionImageBytes); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{"success": false, "message": "invalid multipart upload"})
		return
	}
	file, _, err := r.FormFile("file")
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{"success": false, "message": "image is required"})
		return
	}
	defer file.Close()
	image, err := io.ReadAll(io.LimitReader(file, maxVisionImageBytes+1))
	if err != nil || len(image) == 0 || len(image) > maxVisionImageBytes {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{"success": false, "message": "invalid image"})
		return
	}
	question := strings.TrimSpace(r.FormValue("question"))
	if question == "" {
		question = "이 사진에 무엇이 보이는지 한국어로 간단히 설명해 줘."
	}

	baseURL, apiKey, model := visionProviderConfig(ctx)
	client := newOpenAIClient(baseURL, apiKey, model, "", "", "", "", "", "")
	result, err := client.ExplainImage(ctx, image, question)
	if err != nil {
		_ = json.NewEncoder(w).Encode(map[string]any{"success": false, "message": err.Error()})
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"success": true, "result": result})
}

func visionProviderConfig(ctx context.Context) (string, string, string) {
	provider := configuredProvider(ctx)
	baseURL := aiString(ctx, "vision_base_url", "")
	apiKey := aiString(ctx, "vision_api_key", "")
	model := aiString(ctx, "vision_model", "")
	if provider == "openrouter" {
		baseURL = override(baseURL, override(aiString(ctx, "llm_base_url", ""), aiString(ctx, "openrouter_base_url", "https://openrouter.ai/api/v1")))
		apiKey = override(apiKey, override(aiString(ctx, "llm_api_key", ""), aiString(ctx, "openrouter_api_key", "")))
		model = override(model, override(aiString(ctx, "llm_model", ""), aiString(ctx, "compatible_model", "")))
	} else {
		baseURL = override(baseURL, override(aiString(ctx, "llm_base_url", ""), aiString(ctx, "compatible_base_url", "")))
		apiKey = override(apiKey, override(aiString(ctx, "llm_api_key", ""), aiString(ctx, "compatible_api_key", "")))
		model = override(model, override(aiString(ctx, "llm_model", ""), aiString(ctx, "compatible_model", "")))
	}
	return baseURL, apiKey, model
}

func (c *openAIClient) ExplainImage(ctx context.Context, image []byte, question string) (string, error) {
	if c.apiKey == "" || c.model == "" {
		return "", fmt.Errorf("vision API key and model are required")
	}
	dataURL := "data:" + detectImageMediaType(image) + ";base64," + base64.StdEncoding.EncodeToString(image)
	body, _ := json.Marshal(map[string]any{
		"model": c.model,
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "text", "text": visionQuestionText(question)},
				{"type": "image_url", "image_url": map[string]any{"url": dataURL}},
			},
		}},
		"max_tokens": compatibleChatMaxTokens,
	})
	data, err := c.doRequest(ctx, http.MethodPost, "/v1/chat/completions", bytes.NewReader(body), "application/json")
	if err != nil {
		return "", err
	}
	var resp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(data, &resp) != nil || len(resp.Choices) == 0 || strings.TrimSpace(resp.Choices[0].Message.Content) == "" {
		return "", fmt.Errorf("unexpected vision response")
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

func visionQuestionText(question string) string {
	lowered := strings.ToLower(question)
	if strings.Contains(lowered, "json") || strings.Contains(question, "JSON") {
		return question + " Return only valid JSON. Do not use markdown fences or explanatory text."
	}
	return question + " 답은 반드시 자연스러운 한국어로 짧고 명확하게 해 줘."
}

func detectImageMediaType(image []byte) string {
	mediaType := http.DetectContentType(image)
	switch mediaType {
	case "image/jpeg", "image/png", "image/gif", "image/webp":
		return mediaType
	default:
		return "image/jpeg"
	}
}
