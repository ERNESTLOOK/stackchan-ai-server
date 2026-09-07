/*
SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
SPDX-License-Identifier: MIT
*/

package ai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestDeviceMCPInitializeListsAndCallsTools(t *testing.T) {
	var client *deviceMCPClient
	client = newDeviceMCPClient("session-1", func(v any) error {
		b, _ := json.Marshal(v)
		var request struct {
			Payload struct {
				ID     int64          `json:"id"`
				Method string         `json:"method"`
				Params map[string]any `json:"params"`
			} `json:"payload"`
		}
		_ = json.Unmarshal(b, &request)
		var result any = map[string]any{}
		switch request.Payload.Method {
		case "tools/list":
			result = map[string]any{"tools": []map[string]any{{
				"name": "self.camera.take_photo", "description": "take a photo",
				"inputSchema": map[string]any{"type": "object", "properties": map[string]any{"question": map[string]any{"type": "string"}}},
			}}}
		case "tools/call":
			result = map[string]any{"content": []map[string]any{{"type": "text", "text": "사진 설명"}}, "isError": false}
		}
		go client.handle(map[string]any{"jsonrpc": "2.0", "id": request.Payload.ID, "result": result})
		return nil
	})

	if err := client.initialize(context.Background(), "http://camera/explain", "token"); err != nil {
		t.Fatal(err)
	}
	tools := client.openAITools()
	if len(tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(tools))
	}
	name := tools[0]["function"].(map[string]any)["name"]
	if name != "device__self__camera__take_photo" {
		t.Fatalf("tool name = %v", name)
	}
	description := tools[0]["function"].(map[string]any)["description"].(string)
	if !strings.Contains(description, "what Eve can see") {
		t.Fatalf("camera description was not enhanced: %q", description)
	}
	if names := client.toolNames(); len(names) != 1 || names[0] != "self.camera.take_photo" {
		t.Fatalf("toolNames=%v", names)
	}
	if !client.hasTool("self.camera.take_photo") || client.hasTool("self.robot.set_head_angles") {
		t.Fatal("hasTool returned unexpected availability")
	}
	result, err := client.call(context.Background(), name.(string), map[string]any{"question": "뭐가 보여?"})
	if err != nil || result != "사진 설명" {
		t.Fatalf("call result=%q err=%v", result, err)
	}
}

func TestEmotionForText(t *testing.T) {
	tests := map[string]string{
		"교수님, 좋아! 내가 도와줄게": "happy",
		"미안해, 속상했겠다":       "sad",
		"흥, 나 삐질 거야":       "doubtful",
		"나 진짜 화나고 짜증나":     "angry",
		"어라, 왜 그런 걸까?":     "doubtful",
		"지금 확인할게":          "neutral",
	}
	for text, want := range tests {
		if got := emotionForText(text); got != want {
			t.Errorf("emotionForText(%q)=%q, want %q", text, got, want)
		}
	}
}

func TestReactionForEmotionUsesSafeMotionRange(t *testing.T) {
	for _, emotion := range []string{"neutral", "happy", "angry", "sad", "sleepy", "doubtful"} {
		reaction := reactionForEmotion(emotion)
		if reaction.Yaw < -20 || reaction.Yaw > 20 || reaction.Pitch < 0 || reaction.Pitch > 20 {
			t.Fatalf("%s reaction out of normal range: %+v", emotion, reaction)
		}
		if reaction.Red < 0 || reaction.Red > 168 || reaction.Green < 0 || reaction.Green > 168 || reaction.Blue < 0 || reaction.Blue > 168 {
			t.Fatalf("%s LED out of safe range: %+v", emotion, reaction)
		}
		if reaction.LEDPattern == "" || reaction.LEDSpeed <= 0 {
			t.Fatalf("%s missing LED behavior: %+v", emotion, reaction)
		}
		for _, step := range ledPatternSteps(reaction) {
			if step.Red < 0 || step.Red > 168 || step.Green < 0 || step.Green > 168 || step.Blue < 0 || step.Blue > 168 {
				t.Fatalf("%s LED step out of safe range: %+v", emotion, step)
			}
		}
	}
}

func TestLEDPatternsReflectEmotionTempo(t *testing.T) {
	if reactionForEmotion("angry").LEDSpeed >= reactionForEmotion("happy").LEDSpeed {
		t.Fatal("angry/excited LED pulse should be faster than happy sparkle")
	}
	if reactionForEmotion("sad").LEDSpeed <= reactionForEmotion("angry").LEDSpeed {
		t.Fatal("sad LED breathe should be slower than excited pulse")
	}
	if len(ledPatternSteps(reactionForEmotion("sad"))) < 5 {
		t.Fatal("sad breathing pattern should have gradual steps")
	}
}

func TestAutomaticVisionContextUsesCameraForVisualQuestion(t *testing.T) {
	var client *deviceMCPClient
	called := false
	client = newDeviceMCPClient("session-vision", func(v any) error {
		b, _ := json.Marshal(v)
		var request struct {
			Payload struct {
				ID     int64  `json:"id"`
				Method string `json:"method"`
			} `json:"payload"`
		}
		_ = json.Unmarshal(b, &request)
		if request.Payload.Method == "tools/call" {
			called = true
			go client.handle(map[string]any{
				"jsonrpc": "2.0",
				"id":      request.Payload.ID,
				"result": map[string]any{
					"content": []map[string]any{{"type": "text", "text": "손가락이 화면 왼쪽에 있습니다."}},
					"isError": false,
				},
			})
		}
		return nil
	})
	client.nameMap = map[string]string{openAIDeviceToolName("self.camera.take_photo"): "self.camera.take_photo"}

	history := appendAutomaticVisionContext(context.Background(), []chatMessage{{Role: "user", Content: "손가락이 어디 있어?"}}, "손가락이 어디 있어?", client)
	if !called {
		t.Fatal("automatic vision did not call the camera")
	}
	if len(history) != 2 || !strings.Contains(history[1].Content, "손가락이 화면 왼쪽") {
		t.Fatalf("history=%#v", history)
	}
}

func TestAutomaticVisionContextSkipsNonVisualQuestion(t *testing.T) {
	client := newDeviceMCPClient("session-vision", func(any) error {
		t.Fatal("camera should not be called")
		return nil
	})
	client.nameMap = map[string]string{openAIDeviceToolName("self.camera.take_photo"): "self.camera.take_photo"}
	history := appendAutomaticVisionContext(context.Background(), []chatMessage{{Role: "user", Content: "오늘 일정 알려줘"}}, "오늘 일정 알려줘", client)
	if len(history) != 1 {
		t.Fatalf("history=%#v", history)
	}
}

func TestParseFaceObservationAndAngles(t *testing.T) {
	obs, ok := parseFaceObservation(`{"face":true,"horizontal":"left","vertical":"up","confidence":0.86}`)
	if !ok || !obs.Face || obs.Horizontal != "left" || obs.Vertical != "up" {
		t.Fatalf("obs=%+v ok=%t", obs, ok)
	}
	yaw, pitch := faceContactAngles(obs)
	if yaw != -10 || pitch != 14 {
		t.Fatalf("yaw=%d pitch=%d", yaw, pitch)
	}
	obs, ok = parseFaceObservation("```json\n{\"face\":true,\"horizontal\":\"right\",\"vertical\":\"down\",\"confidence\":0.9}\n```")
	if !ok {
		t.Fatal("failed to extract JSON from fenced response")
	}
	yaw, pitch = faceContactAngles(obs)
	if yaw != 10 || pitch != 3 {
		t.Fatalf("yaw=%d pitch=%d", yaw, pitch)
	}
}
