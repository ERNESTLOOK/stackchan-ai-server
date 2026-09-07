/*
SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
SPDX-License-Identifier: MIT
*/

// Provider abstraction for the realtime LLM/TTS backend.
//
// ws_simulator.go talks to whatever LLM provider is configured via this
// interface. Audio is normalized to int16 PCM (16kHz in, 24kHz out) and
// callbacks are kept provider-agnostic, so any new backend (OpenAI Realtime,
// Gemini Live, etc.) only needs to map its wire protocol → these callbacks.
package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/gogf/gf/v2/frame/g"
)

// deviceProfile overrides global behaviour for one physical device. API keys
// remain global, so profile JSON can safely be stored in add-on options.
type deviceProfile struct {
	Provider            string `json:"provider"`
	SystemPrompt        string `json:"system_prompt"`
	OpenAIRealtimeModel string `json:"openai_realtime_model"`
	OpenAITTSVoice      string `json:"openai_tts_voice"`
	GeminiModel         string `json:"gemini_model"`
	GeminiVoice         string `json:"gemini_voice"`
	CompatibleModel     string `json:"compatible_model"`
	CompatibleSTTModel  string `json:"compatible_stt_model"`
	CompatibleTTSModel  string `json:"compatible_tts_model"`
	CompatibleTTSVoice  string `json:"compatible_tts_voice"`
}

func deviceProfileFor(ctx context.Context, deviceID string) deviceProfile {
	if raw, ok := storedSetting("device_profiles"); ok {
		if raw == "" {
			return deviceProfile{}
		}
		return deviceProfileFromJSON(raw, deviceID)
	}
	return deviceProfileFromJSON(configuredDeviceProfiles(ctx), deviceID)
}

func configuredDeviceProfiles(ctx context.Context) string {
	if value := g.Cfg().MustGet(ctx, "ai.device_profiles", "").String(); value != "" {
		return value
	}
	encoded := g.Cfg().MustGet(ctx, "ai.device_profiles_b64", "").String()
	if encoded == "" {
		return ""
	}
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return ""
	}
	return string(raw)
}

func deviceProfileFromJSON(raw, deviceID string) deviceProfile {
	if raw == "" || deviceID == "" {
		return deviceProfile{}
	}
	profiles := map[string]deviceProfile{}
	if json.Unmarshal([]byte(raw), &profiles) != nil {
		return deviceProfile{}
	}
	return profiles[deviceID]
}

func override(value, fallback string) string {
	if value != "" {
		return value
	}
	return fallback
}

func configuredProvider(ctx context.Context) string {
	return aiString(ctx, "provider", "openai")
}

func globalSystemPrompt(ctx context.Context) string {
	const fallback = "You are StackChan, a friendly desktop robot assistant."
	if prompt, ok := storedSetting("system_prompt"); ok {
		if prompt != "" {
			return prompt
		}
		return fallback
	}
	encoded := g.Cfg().MustGet(ctx, "ai.system_prompt_b64", "").String()
	if raw, err := base64.StdEncoding.DecodeString(encoded); err == nil && len(raw) > 0 {
		return string(raw)
	}
	return fallback
}

// RealtimeSession is the wire-protocol-agnostic surface ws_simulator depends on.
// One session per device WebSocket connection.
type RealtimeSession interface {
	// AppendAudio sends a PCM16 16kHz mono chunk to the model's input buffer.
	AppendAudio(pcm []int16) error
	// CommitAudio explicitly flushes the input buffer. May be a no-op for
	// providers that always rely on server VAD (e.g. Gemini Live).
	CommitAudio() error
	// CancelResponse interrupts an in-progress reply (wake word / abort).
	CancelResponse() error
	// Close shuts down the underlying connection.
	Close()
}

// ServerVADRequired marks HTTP-style providers that cannot detect the end of
// an utterance themselves. The stock Xiaozhi firmware starts an audio stream
// but does not always send listen:stop, so wsSession must commit after trailing
// silence for these providers.
type ServerVADRequired interface {
	RequiresServerVAD() bool
}

// PlaybackStateAware lets a provider postpone asynchronous announcements
// until the device has physically drained its audio queue.
type PlaybackStateAware interface {
	SetPlaybackBusy(bool)
	InterruptPlayback()
}

// AsyncDeliveryAware gates persisted result announcements until the Xiaozhi
// hello handshake has completed on a newly connected device.
type AsyncDeliveryAware interface {
	SetDeliveryReady(bool)
}

// RealtimeCallbacks bundles the events ws_simulator wants from the provider.
// All callbacks are invoked from the provider's read goroutine — keep them
// quick and lock-free where possible.
type RealtimeCallbacks struct {
	OnSTT   func(string)  // user speech transcript
	OnText  func(string)  // model text reply (logging only)
	OnAudio func([]int16) // 24kHz PCM chunk to play back to the device
	OnStart func()        // model began speaking
	OnStop  func()        // model finished speaking
	OnClose func()        // provider connection ended (any reason — error or normal)
}

// dialProvider selects the configured backend and opens a
// session. Provider-specific config keys live under ai.* in the generated
// config.yaml (written by addon/run.sh from /data/options.json).
func dialProvider(
	ctx context.Context,
	deviceID string,
	ha *haWSClient,
	cb RealtimeCallbacks,
) (RealtimeSession, error) {
	p := deviceProfileFor(ctx, deviceID)
	provider := override(p.Provider, configuredProvider(ctx))
	sysPrompt := override(p.SystemPrompt, globalSystemPrompt(ctx))
	sysPrompt += deviceCapabilityPrompt()
	if memory, err := conversationContext(ctx, deviceID); err == nil {
		sysPrompt += memory
	} else {
		g.Log().Warning(ctx, "[HISTORY] could not load conversation context; continuing without history")
	}

	switch provider {
	case "openai_compatible", "tokenhub", "openrouter":
		return dialCompatibleSession(ctx, compatibleConfigFor(ctx, p, provider, sysPrompt), ha, cb)

	case "gemini":
		apiKey := aiString(ctx, "gemini_api_key", "")
		if apiKey == "" {
			return nil, fmt.Errorf("ai.gemini_api_key is required when provider=gemini")
		}
		model := override(p.GeminiModel, aiString(ctx, "gemini_model", "gemini-2.5-flash-preview-native-audio-dialog"))
		voice := override(p.GeminiVoice, aiString(ctx, "gemini_voice", "Aoede"))
		return dialGeminiSession(ctx, apiKey, model, voice, sysPrompt, ha, cb)

	case "openai", "":
		apiKey := aiString(ctx, "openai_api_key", "")
		if apiKey == "" {
			return nil, fmt.Errorf("ai.openai_api_key is required when provider=openai")
		}
		model := override(p.OpenAIRealtimeModel, aiString(ctx, "openai_realtime_model", "gpt-realtime"))
		voice := override(p.OpenAITTSVoice, aiString(ctx, "openai_tts_voice", "alloy"))
		return dialOpenAIRealtimeSession(ctx, deviceID, apiKey, model, voice, sysPrompt, ha, cb)

	default:
		return nil, fmt.Errorf("unknown ai.provider %q", provider)
	}
}

func deviceCapabilityPrompt() string {
	return "\n\nDevice capability rules:\n" +
		"- Device tools are high-level StackChan community-style actions such as see, face, move, nod, shake, status, health, and sense.\n" +
		"- Camera/vision is opt-in: call stackchan_see only when the user explicitly asks Eve to use the camera, take a photo, or look at what is physically in front of her. Never infer visual facts without a camera result.\n" +
		"- Motion tools are command-only: call stackchan_move, stackchan_nod, or stackchan_shake only when the user explicitly asks for that physical movement. Do not use them for idle liveliness; the firmware's own idle/blink/breath/touch/speaking loops own that behavior.\n" +
		"- Use stackchan_face only as a short expression cue when it materially helps the interaction; do not micromanage LEDs or servos.\n" +
		"- If a fact was not heard from the user, remembered from conversation history, or returned by a tool, say briefly that it needs confirmation. Do not invent observations, capabilities, or events.\n" +
		"- For voice replies, answer in natural Korean in one or two short sentences, call the user 교수님, and ask at most one question."
}

func compatibleConfigFor(ctx context.Context, profile deviceProfile, provider, sysPrompt string) compatibleConfig {
	compatibleBaseURL := aiString(ctx, "compatible_base_url", "")
	compatibleAPIKey := aiString(ctx, "compatible_api_key", "")
	llmBaseURL, llmAPIKey := compatibleBaseURL, compatibleAPIKey
	if provider == "tokenhub" {
		llmBaseURL = aiString(ctx, "tokenhub_base_url", llmBaseURL)
		llmAPIKey = aiString(ctx, "tokenhub_api_key", llmAPIKey)
	}
	if provider == "openrouter" {
		llmBaseURL = aiString(ctx, "openrouter_base_url", "https://openrouter.ai/api/v1")
		llmAPIKey = aiString(ctx, "openrouter_api_key", llmAPIKey)
	}
	// Stage-specific values override the legacy compatible endpoint. This
	// enables e.g. domestic STT + TokenHub LLM + local TTS without breaking
	// existing single-endpoint configurations.
	sttBaseURL := override(aiString(ctx, "stt_base_url", ""), compatibleBaseURL)
	sttAPIKey := override(aiString(ctx, "stt_api_key", ""), compatibleAPIKey)
	ttsBaseURL := override(aiString(ctx, "tts_base_url", ""), compatibleBaseURL)
	ttsAPIKey := override(aiString(ctx, "tts_api_key", ""), compatibleAPIKey)
	llmBaseURL = override(aiString(ctx, "llm_base_url", ""), llmBaseURL)
	llmAPIKey = override(aiString(ctx, "llm_api_key", ""), llmAPIKey)
	sttModel := override(profile.CompatibleSTTModel, override(aiString(ctx, "stt_model", ""), aiString(ctx, "compatible_stt_model", "whisper-1")))
	llmModel := override(profile.CompatibleModel, override(aiString(ctx, "llm_model", ""), aiString(ctx, "compatible_model", "")))
	ttsModel := override(profile.CompatibleTTSModel, override(aiString(ctx, "tts_model", ""), aiString(ctx, "compatible_tts_model", "tts-1")))
	voice := override(profile.CompatibleTTSVoice, override(aiString(ctx, "tts_voice", ""), aiString(ctx, "compatible_tts_voice", "alloy")))
	ttsInstructions := aiString(ctx, "tts_instructions", "")
	ttsVolumeGain := min(5.0, max(0.1, aiFloat(ctx, "tts_volume_gain", 1.0)))
	return compatibleConfig{
		STTBaseURL:      sttBaseURL,
		STTAPIKey:       sttAPIKey,
		STTModel:        sttModel,
		LLMBaseURL:      llmBaseURL,
		LLMAPIKey:       llmAPIKey,
		LLMModel:        llmModel,
		TTSBaseURL:      ttsBaseURL,
		TTSAPIKey:       ttsAPIKey,
		TTSModel:        ttsModel,
		Voice:           voice,
		TTSInstructions: ttsInstructions,
		TTSVolumeGain:   ttsVolumeGain,
		Prompt:          sysPrompt,
	}
}
