/*
SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
SPDX-License-Identifier: MIT
*/

package ai

// The OpenAI-compatible pipeline intentionally waits for listen:stop before
// uploading audio. Most compatible endpoints expose HTTP/SSE chat APIs rather
// than the bidirectional Realtime WebSocket protocol, so pretending otherwise
// would make interruption and latency behaviour unreliable.

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/gogf/gf/v2/frame/g"
	"github.com/gogf/gf/v2/os/gctx"
)

type compatibleConfig struct {
	STTBaseURL, STTAPIKey, STTModel        string
	LLMBaseURL, LLMAPIKey, LLMModel        string
	TTSBaseURL, TTSAPIKey, TTSModel, Voice string
	TTSInstructions                        string
	TTSVolumeGain                          float64
	Prompt                                 string
}

type compatibleSession struct {
	sttClient     *openAIClient
	llmClient     *openAIClient
	ttsClient     *openAIClient
	ha            *haWSClient
	cb            RealtimeCallbacks
	ctx           context.Context
	cancel        context.CancelFunc
	mu            sync.Mutex
	pcm           []int16
	history       []chatMessage
	turnCancel    context.CancelFunc
	ttsVolumeGain float64
	deviceTools   *deviceMCPClient
	closed        bool
}

func dialCompatibleSession(ctx context.Context, cfg compatibleConfig, ha *haWSClient, cb RealtimeCallbacks) (RealtimeSession, error) {
	if cfg.STTBaseURL == "" || cfg.STTAPIKey == "" || cfg.STTModel == "" {
		return nil, fmt.Errorf("STT base URL, API key, and model are required")
	}
	if cfg.LLMBaseURL == "" || cfg.LLMAPIKey == "" || cfg.LLMModel == "" {
		return nil, fmt.Errorf("LLM base URL, API key, and model are required")
	}
	if cfg.TTSBaseURL == "" || cfg.TTSAPIKey == "" || cfg.TTSModel == "" {
		return nil, fmt.Errorf("TTS base URL, API key, and model are required")
	}
	childCtx, cancel := context.WithCancel(ctx)
	return &compatibleSession{
		sttClient: newOpenAIClient(cfg.STTBaseURL, cfg.STTAPIKey, "", cfg.STTModel, "", "", "", ""),
		llmClient: newOpenAIClient(cfg.LLMBaseURL, cfg.LLMAPIKey, cfg.LLMModel, "", "", "", "", cfg.Prompt),
		ttsClient: newOpenAIClient(cfg.TTSBaseURL, cfg.TTSAPIKey, "", "", cfg.TTSModel, cfg.Voice, cfg.TTSInstructions, ""),
		ha:        ha, cb: cb, ctx: childCtx, cancel: cancel, ttsVolumeGain: cfg.TTSVolumeGain,
	}, nil
}

func (s *compatibleSession) AppendAudio(pcm []int16) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return context.Canceled
	}
	s.pcm = append(s.pcm, pcm...)
	return nil
}

func (s *compatibleSession) RequiresServerVAD() bool { return true }

func (s *compatibleSession) CommitAudio() error {
	s.mu.Lock()
	if s.closed || len(s.pcm) == 0 {
		s.mu.Unlock()
		return nil
	}
	pcm := append([]int16(nil), s.pcm...)
	s.pcm = s.pcm[:0]
	if s.turnCancel != nil {
		s.turnCancel()
	}
	turnCtx, turnCancel := context.WithCancel(s.ctx)
	s.turnCancel = turnCancel
	s.mu.Unlock()
	go s.completeTurn(turnCtx, pcm)
	return nil
}

func (s *compatibleSession) completeTurn(ctx context.Context, pcm []int16) {
	text, err := s.sttClient.Transcribe(ctx, pcmToWAV(pcm, 16000))
	if err != nil {
		g.Log().Warningf(gctx.New(), "[COMPAT] transcription: %v", err)
		return
	}
	if text == "" {
		return
	}
	if s.cb.OnSTT != nil {
		s.cb.OnSTT(text)
	}

	s.mu.Lock()
	history := append([]chatMessage(nil), s.history...)
	history = append(history, chatMessage{Role: "user", Content: text})
	deviceTools := s.deviceTools
	s.mu.Unlock()
	history = appendAutomaticVisionContext(ctx, history, text, deviceTools)
	reply, err := s.llmClient.Chat(ctx, history, s.ha, deviceTools)
	if err != nil {
		g.Log().Warningf(gctx.New(), "[COMPAT] chat: %v", err)
		return
	}
	if reply == "" {
		return
	}
	if s.cb.OnText != nil {
		s.cb.OnText(reply)
	}
	if s.cb.OnStart != nil {
		s.cb.OnStart()
	}
	pcmReply, err := s.ttsClient.Speak(ctx, reply)
	if err != nil {
		g.Log().Warningf(gctx.New(), "[COMPAT] TTS: %v", err)
	} else if len(pcmReply) > 0 && s.cb.OnAudio != nil {
		clipped := applyPCMVolume(pcmReply, s.ttsVolumeGain)
		g.Log().Infof(gctx.New(), "[COMPAT] TTS volume gain=%.2f samples=%d clipped=%d", s.ttsVolumeGain, len(pcmReply), clipped)
		s.cb.OnAudio(pcmReply)
	}
	if s.cb.OnStop != nil {
		s.cb.OnStop()
	}

	s.mu.Lock()
	s.history = append(history, chatMessage{Role: "assistant", Content: reply})
	s.mu.Unlock()
}

func appendAutomaticVisionContext(ctx context.Context, history []chatMessage, text string, device *deviceMCPClient) []chatMessage {
	if !shouldUseAutomaticVision(text, device) {
		return history
	}
	result, err := device.callTool(ctx, "self.camera.take_photo", map[string]any{"question": text})
	if err != nil {
		g.Log().Warningf(gctx.New(), "[VISION] automatic camera failed: %v", err)
		history = append(history, chatMessage{Role: "system", Content: "Eve attempted to use the camera for this user request, but the camera call failed: " + err.Error()})
		return history
	}
	g.Log().Infof(gctx.New(), "[VISION] automatic camera result=%s", result)
	return append(history, chatMessage{Role: "system", Content: "Eve already used the camera for the user's latest visual request. Camera observation: " + result + "\nAnswer the user from this observation in short, natural Korean. Do not call the camera again unless the observation is insufficient."})
}

func shouldUseAutomaticVision(text string, device *deviceMCPClient) bool {
	if device == nil || !device.hasTool("self.camera.take_photo") {
		return false
	}
	lowered := strings.ToLower(text)
	for _, word := range []string{
		"카메라", "사진", "이미지", "비전", "눈으로", "앞에", "보여", "보이나", "보이니", "봐줘", "봐 줘", "찾아", "어디", "손가락",
		"camera", "photo", "image", "vision", "look", "see", "find", "where", "finger",
	} {
		if strings.Contains(lowered, word) {
			return true
		}
	}
	return false
}

func (s *compatibleSession) SetDeviceTools(tools *deviceMCPClient) {
	s.mu.Lock()
	s.deviceTools = tools
	s.mu.Unlock()
}

func applyPCMVolume(pcm []int16, gain float64) int {
	if gain <= 0 || gain == 1 {
		return 0
	}
	clipped := 0
	for i, sample := range pcm {
		scaled := int(float64(sample) * gain)
		if scaled > 32767 {
			scaled = 32767
			clipped++
		} else if scaled < -32768 {
			scaled = -32768
			clipped++
		}
		pcm[i] = int16(scaled)
	}
	return clipped
}

func (s *compatibleSession) CancelResponse() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.turnCancel != nil {
		s.turnCancel()
		s.turnCancel = nil
	}
	return nil
}

func (s *compatibleSession) Close() {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	s.closed = true
	if s.turnCancel != nil {
		s.turnCancel()
	}
	s.mu.Unlock()
	s.cancel()
}
