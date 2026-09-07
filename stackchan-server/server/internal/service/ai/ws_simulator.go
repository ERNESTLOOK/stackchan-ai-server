/*
SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
SPDX-License-Identifier: MIT
*/

// Package ai implements a Xiaozhi WebSocket protocol v3 simulator backed by
// the OpenAI Realtime API (gpt-realtime).
//
// Audio path:
//
//	device OPUS (16kHz) → PCM → Realtime WS input buffer
//	                              ↓  server VAD detects speech end
//	                         Realtime WS output (PCM 24kHz, streaming)
//	                              ↓
//	                    opusStreamEncoder → frameQueue (chan)
//	                              ↓
//	                    pacingLoop (60ms ticker) → device OPUS frames
//
// pacingLoop paces frame delivery at exactly one frame per 60ms, preventing
// burst delivery that causes audio stuttering on the device.
// A nil sentinel in frameQueue signals the end of a response; pacingLoop
// sends tts:stop only after all queued frames have been delivered.
package ai

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gogf/gf/v2/frame/g"
	"github.com/gogf/gf/v2/os/gctx"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	opus "gopkg.in/hraban/opus.v2"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

const frameQueueSize = 600 // ~36 seconds of audio headroom

const (
	serverVADThreshold   = int64(400 * 400)
	serverVADSilenceTime = 360 * time.Millisecond
)

const (
	loudStartlePeakThreshold = 22000
	loudStartleRMSThreshold  = int64(9000 * 9000)
	emotionLEDGradientSteps  = 4
	emotionLEDSleepAfter     = 60 * time.Second
	emotionLEDHold           = 24 * time.Second
)

type deviceReaction struct {
	Yaw, Pitch int
	Red        int
	Green      int
	Blue       int
	LEDPattern string
	LEDSpeed   time.Duration
}

type emotionLEDProfile struct {
	Stops    [][3]int
	Interval time.Duration
}

type wsSession struct {
	conn      *websocket.Conn
	deviceID  string
	sessionID string
	activity  *conversationActivity
	rt        RealtimeSession
	opusDec   *opus.Decoder // device input decoder (16kHz, reset per utterance)

	mu               sync.Mutex         // protects opusEnc and isListening
	actionMu         sync.Mutex         // serialises head/LED sequences sent to fragile device firmware
	opusEnc          *opusStreamEncoder // non-nil only while model is speaking
	isListening      bool
	playbackStarted  bool
	prebufferFrames  int
	prebufferMaxWait time.Duration
	prebufferTimer   *time.Timer

	// frameQueue carries encoded OPUS frames to pacingLoop.
	// A nil entry is a sentinel meaning "response ended — send tts:stop".
	frameQueue chan []byte

	writeMu              sync.Mutex // serialises WebSocket writes
	providerClosed       int32      // atomic: 1 when OnClose triggered conn.Close()
	listenStopped        time.Time  // latency baseline for the current user turn
	firstAudioLogged     int32
	inputAudioLogged     int32
	inputDecodeErrors    int32
	inputFrames          int
	inputSamples         int
	serverVAD            bool
	vadHeardSpeech       bool
	vadSilenceSamples    int
	deviceMCP            *deviceMCPClient
	responseEmotion      string
	responseLEDEmotion   string
	responseEmotionAt    time.Time
	lastInteraction      time.Time
	preserveIdleOnSilent bool
	lastFaceContact      time.Time
	faceContactBusy      bool
	autonomousBusy       bool
	autonomousCount      int64
	loudStartleBusy      bool
	lastLoudStartle      time.Time
	ledGeneration        int64
}

// HandleWS upgrades the connection and runs a Xiaozhi v3 protocol session
// backed by OpenAI Realtime API.
func HandleWS(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithCancel(gctx.New())
	defer cancel()

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		g.Log().Errorf(ctx, "ws upgrade: %v", err)
		return
	}

	deviceID := r.Header.Get("Device-Id")
	provider := override(deviceProfileFor(ctx, deviceID).Provider, configuredProvider(ctx))
	var ha *haWSClient
	haEnabled, haURL, haToken := homeAssistantConnection(ctx)
	if haEnabled {
		g.Log().Infof(ctx, "[WS] device=%s connecting HA at %s", deviceID, haURL)
		ha, err = dialHAWebSocket(haURL, haToken)
		if err != nil {
			g.Log().Warningf(ctx, "[WS] device=%s HA connect failed: %v", deviceID, err)
			conn.Close()
			return
		}
		g.Log().Infof(ctx, "[WS] device=%s HA connected", deviceID)
	} else {
		g.Log().Infof(ctx, "[WS] device=%s running without Home Assistant", deviceID)
	}

	opusDec, err := newOpusDecoder()
	if err != nil {
		g.Log().Errorf(ctx, "[WS] device=%s opus decoder init: %v", deviceID, err)
		conn.Close()
		closeHAClient(ha)
		return
	}

	s := &wsSession{
		conn:               conn,
		deviceID:           deviceID,
		sessionID:          uuid.New().String(),
		activity:           newConversationActivity(time.Duration(conversationIdleSeconds(ctx))*time.Second, time.Now()),
		opusDec:            opusDec,
		frameQueue:         make(chan []byte, frameQueueSize),
		prebufferFrames:    max(0, aiInt(ctx, "audio_prebuffer_ms", 300)/frameDurationMs),
		prebufferMaxWait:   time.Duration(max(0, aiInt(ctx, "audio_prebuffer_max_wait_ms", 900))) * time.Millisecond,
		responseEmotion:    "neutral",
		responseLEDEmotion: "neutral",
		lastInteraction:    time.Now(),
	}
	s.deviceMCP = newDeviceMCPClient(s.sessionID, s.sendJSON)

	// Wire provider callbacks → device WebSocket writes. Same callbacks for any
	// backend (OpenAI Realtime, Gemini Live, ...) — see provider.go.
	cb := RealtimeCallbacks{
		OnClose: func() {
			// Provider session ended (error or normal). Close the device
			// connection so run() exits and the device reconnects cleanly
			// rather than waiting forever for a response that won't arrive.
			g.Log().Infof(ctx, "[WS] device=%s provider closed, dropping device connection", deviceID)
			atomic.StoreInt32(&s.providerClosed, 1)
			s.conn.Close()
		},
		OnSTT: func(text string) {
			if err := appendConversation(ctx, deviceID, s.sessionID, provider, "user", text); err != nil {
				g.Log().Warning(ctx, "[HISTORY] could not save user transcript")
			}
			s.alignFaceIfDue(ctx, "stt")
			s.logTurnLatency(ctx, "stt")
			_ = s.sendJSON(map[string]any{"type": "stt", "text": text})
		},

		OnText: func(text string) {
			if err := appendConversation(ctx, deviceID, s.sessionID, provider, "assistant", text); err != nil {
				g.Log().Warning(ctx, "[HISTORY] could not save assistant transcript")
			}
			s.logTurnLatency(ctx, "llm")
			g.Log().Infof(ctx, "[WS] device=%s LLM: %q", deviceID, text)
			s.mu.Lock()
			s.responseEmotion = emotionForText(text)
			s.responseLEDEmotion = emotionLEDForText(text)
			s.responseEmotionAt = time.Now()
			s.mu.Unlock()
		},

		OnAudio: func(pcm []int16) { // encode and enqueue; pacingLoop sends at 60ms
			if !s.activity.responding(time.Now()) {
				return
			}
			if atomic.CompareAndSwapInt32(&s.firstAudioLogged, 0, 1) {
				s.logTurnLatency(ctx, "first_audio")
			}
			s.mu.Lock()
			enc := s.opusEnc
			s.mu.Unlock()
			if enc == nil {
				return
			}
			frames, err := enc.Encode(pcm)
			if err != nil {
				g.Log().Warningf(ctx, "[WS] device=%s encode: %v", deviceID, err)
				return
			}
			for _, frame := range frames {
				select {
				case s.frameQueue <- frame:
				default:
					g.Log().Warningf(ctx, "[WS] device=%s frame queue full, dropping frame", deviceID)
				}
			}
			s.startPlayback(ctx, false)
		},

		OnStart: func() {
			if !s.activity.responding(time.Now()) {
				return
			}
			if aware, ok := s.rt.(PlaybackStateAware); ok {
				aware.SetPlaybackBusy(true)
			}
			s.logTurnLatency(ctx, "tts_start")
			g.Log().Infof(ctx, "[WS] device=%s TTS start", deviceID)
			s.drainFrameQueue() // clear any leftover frames from previous response
			enc, err := newOpusStreamEncoder()
			if err != nil {
				g.Log().Warningf(ctx, "[WS] device=%s encoder init: %v", deviceID, err)
				s.finishTurnWithoutPlayback(ctx, "encoder_init_failed")
				return
			}
			s.mu.Lock()
			s.opusEnc = enc
			s.playbackStarted = false
			s.mu.Unlock()
			if s.prebufferMaxWait > 0 {
				s.mu.Lock()
				s.prebufferTimer = time.AfterFunc(s.prebufferMaxWait, func() { s.startPlayback(ctx, true) })
				s.mu.Unlock()
			}
			if s.prebufferFrames == 0 {
				s.startPlayback(ctx, true)
			}
		},

		OnStop: func() { // flush encoder tail, push nil sentinel; pacingLoop sends tts:stop
			g.Log().Infof(ctx, "[WS] device=%s TTS response done, draining queue", deviceID)
			s.mu.Lock()
			enc := s.opusEnc
			s.mu.Unlock()
			if enc == nil {
				s.finishTurnWithoutPlayback(ctx, "tts_no_audio")
				return
			}
			// Flush remaining PCM that didn't fill a complete 60ms frame.
			for _, frame := range enc.Flush() {
				select {
				case s.frameQueue <- frame:
				default:
				}
			}
			select {
			case s.frameQueue <- nil: // pacingLoop confirms physical playback completion
			case <-ctx.Done():
				return
			}
			s.startPlayback(ctx, true)
			s.mu.Lock()
			s.opusEnc = nil
			s.mu.Unlock()
		},

		OnNoSpeech: func() {
			s.mu.Lock()
			s.preserveIdleOnSilent = true
			s.mu.Unlock()
		},

		OnIdle: func() {
			s.finishTurnWithoutPlayback(ctx, "provider_idle")
		},
	}

	rt, err := dialProvider(ctx, deviceID, ha, cb)
	if err != nil {
		g.Log().Errorf(ctx, "[WS] device=%s provider connect: %v", deviceID, err)
		conn.Close()
		closeHAClient(ha)
		return
	}
	s.rt = rt
	if aware, ok := rt.(deviceToolAware); ok {
		aware.SetDeviceTools(s.deviceMCP)
	}
	if providerVAD, ok := rt.(ServerVADRequired); ok {
		s.serverVAD = providerVAD.RequiresServerVAD()
	}
	defer func() {
		s.mu.Lock()
		if s.prebufferTimer != nil {
			s.prebufferTimer.Stop()
		}
		s.mu.Unlock()
	}()
	g.Log().Infof(ctx, "[WS] device=%s realtime session ready provider=%s", deviceID, provider)

	go s.pacingLoop(ctx)
	go s.pingLoop(ctx)
	go s.idleLoop(ctx)
	go s.autonomousLoop(ctx)
	go s.emotionLEDLoop(ctx)
	s.run(ctx)
	cancel()

	g.Log().Infof(ctx, "[WS] device=%s session closed", deviceID)
	rt.Close()
	closeHAClient(ha)
}

func closeHAClient(ha *haWSClient) {
	if ha != nil {
		ha.Close()
	}
}

// pacingLoop delivers OPUS frames to the device at a steady 60ms per frame.
// A nil frame is a sentinel: send tts:stop and resume idle.
func (s *wsSession) pacingLoop(ctx context.Context) {
	ticker := time.NewTicker(frameDurationMs * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if !s.isPlaybackStarted() {
				continue
			}
			select {
			case frame := <-s.frameQueue:
				if frame == nil {
					// All frames delivered — tell device TTS is done.
					_ = s.sendJSON(map[string]any{"type": "tts", "state": "stop"})
					s.mu.Lock()
					s.playbackStarted = false
					preserveIdle := s.preserveIdleOnSilent
					s.preserveIdleOnSilent = false
					s.lastInteraction = time.Now()
					s.mu.Unlock()
					if preserveIdle {
						s.activity.noSpeechDone()
					} else {
						s.activity.playbackDone(time.Now())
					}
					if aware, ok := s.rt.(PlaybackStateAware); ok {
						aware.SetPlaybackBusy(false)
					}
				} else {
					s.activity.responding(time.Now())
					_ = s.sendAudio(frame)
				}
			default:
				// Queue empty this tick — nothing to send.
			}
		}
	}
}

func (s *wsSession) isPlaybackStarted() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.playbackStarted
}

// startPlayback waits for a small, configurable jitter buffer before sending
// tts:start. It prevents short upstream delivery gaps from becoming audible.
func (s *wsSession) startPlayback(ctx context.Context, force bool) {
	s.mu.Lock()
	if s.playbackStarted || s.opusEnc == nil || (!force && len(s.frameQueue) < s.prebufferFrames) {
		s.mu.Unlock()
		return
	}
	s.playbackStarted = true
	if s.prebufferTimer != nil {
		s.prebufferTimer.Stop()
		s.prebufferTimer = nil
	}
	s.mu.Unlock()
	s.mu.Lock()
	emotion := s.responseEmotion
	s.mu.Unlock()
	if emotion == "" {
		emotion = "neutral"
	}
	_ = s.sendJSON(map[string]any{"type": "llm", "emotion": emotion})
	_ = s.sendJSON(map[string]any{"type": "tts", "state": "start"})
	go s.reactToEmotion(ctx, emotion)
}

// drainFrameQueue discards all pending frames (called on abort or new response start).
func (s *wsSession) drainFrameQueue() {
	for {
		select {
		case <-s.frameQueue:
		default:
			return
		}
	}
}

// finishTurnWithoutPlayback always releases the stock firmware from its
// listening/thinking state. A filtered empty STT result used to call OnIdle
// without sending any terminal protocol message, leaving the device looking
// dead until it reset the connection itself.
func (s *wsSession) finishTurnWithoutPlayback(ctx context.Context, reason string) {
	s.activity.playbackDone(time.Now())
	s.mu.Lock()
	s.opusEnc = nil
	s.playbackStarted = false
	if s.prebufferTimer != nil {
		s.prebufferTimer.Stop()
		s.prebufferTimer = nil
	}
	s.mu.Unlock()
	s.drainFrameQueue()
	if aware, ok := s.rt.(PlaybackStateAware); ok {
		aware.InterruptPlayback()
	}
	if err := s.sendJSON(map[string]any{"type": "tts", "state": "stop"}); err != nil {
		g.Log().Warningf(ctx, "[WS] device=%s idle reset failed reason=%s: %v", s.deviceID, reason, err)
		return
	}
	g.Log().Infof(ctx, "[WS] device=%s turn ended without playback reason=%s; device returned to idle", s.deviceID, reason)
}

func (s *wsSession) run(ctx context.Context) {
	defer s.conn.Close()
	for {
		msgType, data, err := s.conn.ReadMessage()
		if err != nil {
			// Suppress the "use of closed network connection" error that fires
			// when OnClose intentionally calls conn.Close() — it is expected.
			if atomic.LoadInt32(&s.providerClosed) == 0 {
				g.Log().Infof(ctx, "[WS] device=%s read error: %v", s.deviceID, err)
			}
			return
		}

		if msgType == websocket.BinaryMessage {
			// BinaryProtocol3: [type][reserved][payload_size_hi][payload_size_lo][payload]
			if len(data) < 4 {
				continue
			}
			payloadSize := int(binary.BigEndian.Uint16(data[2:4]))
			if payloadSize == 0 || len(data) < 4+payloadSize {
				continue
			}
			s.mu.Lock()
			listening := s.isListening
			s.mu.Unlock()
			if !listening {
				continue
			}
			pcm, err := decodeOpusFrame(s.opusDec, data[4:4+payloadSize])
			if err != nil {
				if failures := atomic.AddInt32(&s.inputDecodeErrors, 1); failures <= 3 {
					g.Log().Warningf(ctx, "[WS] device=%s input OPUS decode failed count=%d: %v", s.deviceID, failures, err)
				}
				continue
			}
			if atomic.CompareAndSwapInt32(&s.inputAudioLogged, 0, 1) {
				g.Log().Infof(ctx, "[WS] device=%s first input audio frame opus_bytes=%d pcm_samples=%d", s.deviceID, payloadSize, len(pcm))
			}
			s.mu.Lock()
			s.inputFrames++
			s.inputSamples += len(pcm)
			s.mu.Unlock()
			s.maybeReactToLoudSound(ctx, pcm)
			if !s.activity.audio(time.Now(), pcm) {
				return
			}
			_ = s.rt.AppendAudio(pcm)
			if s.observeServerVAD(ctx, pcm) {
				continue
			}
			continue
		}

		var msg map[string]any
		if err := json.Unmarshal(data, &msg); err != nil {
			continue
		}
		msgTypeStr, _ := msg["type"].(string)
		if state, _ := msg["state"].(string); state != "" {
			g.Log().Infof(ctx, "[WS] device=%s recv type=%s state=%s", s.deviceID, msgTypeStr, state)
		} else {
			g.Log().Infof(ctx, "[WS] device=%s recv type=%s", s.deviceID, msgTypeStr)
		}

		switch msgTypeStr {
		case "hello":
			s.handleHello(ctx, msg)
		case "mcp":
			if s.deviceMCP != nil {
				s.deviceMCP.handle(msg["payload"])
			}
		case "listen":
			s.handleListen(ctx, msg)
		case "abort":
			s.activity.playbackDone(time.Now())
			_ = s.rt.CancelResponse()
			s.mu.Lock()
			s.opusEnc = nil
			s.isListening = false
			s.playbackStarted = false
			if s.prebufferTimer != nil {
				s.prebufferTimer.Stop()
				s.prebufferTimer = nil
			}
			s.mu.Unlock()
			s.drainFrameQueue()
			if aware, ok := s.rt.(PlaybackStateAware); ok {
				aware.InterruptPlayback()
			}
			_ = s.sendJSON(map[string]any{"type": "tts", "state": "stop"})
		}
	}
}

func (s *wsSession) handleHello(ctx context.Context, msg map[string]any) {
	sessionID := s.sessionID
	s.activity.wake(time.Now())
	err := s.sendJSON(map[string]any{
		"type":       "hello",
		"transport":  "websocket",
		"session_id": sessionID,
		"audio_params": map[string]any{
			"sample_rate":    serverSampleRate,
			"frame_duration": frameDurationMs,
		},
	})
	if err != nil {
		g.Log().Warningf(ctx, "[WS] device=%s hello send failed: %v", s.deviceID, err)
		return
	}
	if aware, ok := s.rt.(AsyncDeliveryAware); ok {
		aware.SetDeliveryReady(true)
	}
	g.Log().Infof(ctx, "[WS] device=%s session=%s hello OK", s.deviceID, sessionID)
	features, _ := msg["features"].(map[string]any)
	mcpSupported, _ := features["mcp"].(bool)
	if !mcpSupported || s.deviceMCP == nil {
		g.Log().Infof(ctx, "[MCP] device=%s not advertised", s.deviceID)
		return
	}
	visionURL := fmt.Sprintf("http://%s:%d/xiaozhi/vision/explain", aiString(ctx, "local_host", "127.0.0.1"), aiInt(ctx, "local_port", 12800))
	visionToken := aiString(ctx, "vision_token", "")
	go func() {
		if err := s.deviceMCP.initialize(ctx, visionURL, visionToken); err != nil {
			g.Log().Warningf(ctx, "[MCP] device=%s initialize: %v", s.deviceID, err)
			return
		}
		g.Log().Infof(ctx, "[MCP] device=%s ready tools=%d names=%s vision=%s", s.deviceID, len(s.deviceMCP.openAITools()), strings.Join(s.deviceMCP.toolNames(), ","), visionURL)
	}()
}

type faceObservation struct {
	Face       bool    `json:"face"`
	Horizontal string  `json:"horizontal"`
	Vertical   string  `json:"vertical"`
	Confidence float64 `json:"confidence"`
}

func emotionForText(text string) string {
	text = strings.ToLower(text)
	for _, word := range []string{"미안", "슬퍼", "속상", "안타깝", "걱정", "힘들"} {
		if strings.Contains(text, word) {
			return "sad"
		}
	}
	for _, word := range []string{"화나", "화가", "짜증", "싫어", "싫다", "안 돼"} {
		if strings.Contains(text, word) {
			return "angry"
		}
	}
	for _, word := range []string{"깜짝", "놀라", "앗", "헉", "어?", "어!"} {
		if strings.Contains(text, word) {
			return "surprised"
		}
	}
	for _, word := range []string{"왜", "글쎄", "궁금", "이상하", "어라", "정말?", "흥", "삐질"} {
		if strings.Contains(text, word) {
			return "doubtful"
		}
	}
	for _, word := range []string{"하하", "호호", "좋아", "신나", "축하", "기뻐", "귀여", "재밌", "도와줄게"} {
		if strings.Contains(text, word) {
			return "happy"
		}
	}
	return "neutral"
}

func emotionLEDForText(text string) string {
	text = strings.ToLower(text)
	for _, word := range []string{"졸려", "졸리", "피곤", "하품", "잘 자", "잠이"} {
		if strings.Contains(text, word) {
			return "sleepy"
		}
	}
	for _, word := range []string{"부끄", "쑥스", "헤헤", "칭찬"} {
		if strings.Contains(text, word) {
			return "shy"
		}
	}
	for _, word := range []string{"흥", "삐졌", "삐질", "투정", "몰라"} {
		if strings.Contains(text, word) {
			return "pouty"
		}
	}
	emotion := emotionForText(text)
	if emotion == "doubtful" {
		return "curious"
	}
	return emotion
}

func emotionLEDProfileFor(emotion string) emotionLEDProfile {
	switch emotion {
	case "surprised":
		return emotionLEDProfile{Stops: [][3]int{{35, 8, 65}, {168, 96, 160}, {80, 34, 168}, {168, 48, 112}}, Interval: 250 * time.Millisecond}
	case "angry":
		return emotionLEDProfile{Stops: [][3]int{{35, 0, 2}, {168, 0, 15}, {75, 3, 0}, {168, 28, 0}}, Interval: 280 * time.Millisecond}
	case "happy", "laughing":
		return emotionLEDProfile{Stops: [][3]int{{50, 8, 35}, {168, 34, 118}, {168, 105, 148}, {105, 22, 92}}, Interval: 340 * time.Millisecond}
	case "shy":
		return emotionLEDProfile{Stops: [][3]int{{32, 5, 20}, {148, 42, 98}, {168, 92, 126}, {72, 18, 55}}, Interval: 520 * time.Millisecond}
	case "pouty":
		return emotionLEDProfile{Stops: [][3]int{{38, 0, 12}, {168, 12, 48}, {88, 2, 38}, {145, 25, 80}}, Interval: 420 * time.Millisecond}
	case "sad":
		return emotionLEDProfile{Stops: [][3]int{{4, 8, 25}, {12, 32, 96}, {24, 58, 168}, {8, 22, 68}}, Interval: 720 * time.Millisecond}
	case "sleepy":
		return emotionLEDProfile{Stops: [][3]int{{2, 1, 8}, {14, 6, 38}, {36, 16, 82}, {8, 3, 24}}, Interval: 1050 * time.Millisecond}
	case "curious", "thinking", "doubtful":
		return emotionLEDProfile{Stops: [][3]int{{12, 5, 48}, {72, 32, 145}, {24, 88, 168}, {105, 42, 138}}, Interval: 480 * time.Millisecond}
	default:
		return emotionLEDProfile{Stops: [][3]int{{8, 3, 24}, {35, 14, 82}, {70, 28, 132}, {28, 10, 62}}, Interval: 620 * time.Millisecond}
	}
}

func emotionLEDGradientColor(profile emotionLEDProfile, frame int64) [3]int {
	if len(profile.Stops) == 0 {
		return [3]int{}
	}
	cycle := int64(len(profile.Stops) * emotionLEDGradientSteps)
	phase := frame % cycle
	fromIndex := int(phase / emotionLEDGradientSteps)
	toIndex := (fromIndex + 1) % len(profile.Stops)
	step := int(phase % emotionLEDGradientSteps)
	from := profile.Stops[fromIndex]
	to := profile.Stops[toIndex]
	var color [3]int
	for channel := range color {
		color[channel] = (from[channel]*(emotionLEDGradientSteps-step) + to[channel]*step) / emotionLEDGradientSteps
	}
	return color
}

func (s *wsSession) currentEmotionLEDState(now time.Time) string {
	s.mu.Lock()
	listening := s.isListening
	playing := s.playbackStarted || s.opusEnc != nil
	emotion := s.responseLEDEmotion
	emotionAt := s.responseEmotionAt
	lastInteraction := s.lastInteraction
	s.mu.Unlock()
	if listening {
		return "curious"
	}
	if s.activity != nil && s.activity.responseBusy() && !playing {
		return "thinking"
	}
	if !lastInteraction.IsZero() && now.Sub(lastInteraction) >= emotionLEDSleepAfter {
		return "sleepy"
	}
	if emotion != "" && !emotionAt.IsZero() && now.Sub(emotionAt) < emotionLEDHold {
		return emotion
	}
	return "neutral"
}

func (s *wsSession) emotionLEDLoop(ctx context.Context) {
	if !aiBool(ctx, "stackchan_emotion_led_enabled", false) {
		return
	}
	timer := time.NewTimer(500 * time.Millisecond)
	defer timer.Stop()
	lastEmotion := ""
	var frame int64
	for {
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
		}
		device := s.deviceMCP
		if device == nil || !device.hasTool("self.robot.set_led_color") {
			timer.Reset(500 * time.Millisecond)
			continue
		}
		emotion := s.currentEmotionLEDState(time.Now())
		if emotion != lastEmotion {
			lastEmotion = emotion
			frame = 0
		}
		profile := emotionLEDProfileFor(emotion)
		color := emotionLEDGradientColor(profile, frame)
		callCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
		s.actionMu.Lock()
		_, err := device.callTool(callCtx, "self.robot.set_led_color", map[string]any{"red": color[0], "green": color[1], "blue": color[2]})
		s.actionMu.Unlock()
		cancel()
		if err != nil {
			if ctx.Err() == nil {
				g.Log().Warningf(ctx, "[LED] device=%s emotion heartbeat stopped after device error: %v", s.deviceID, err)
			}
			return
		}
		if frame == 0 {
			g.Log().Infof(ctx, "[LED] device=%s emotion=%s heartbeat_ms=%d", s.deviceID, emotion, profile.Interval.Milliseconds())
		}
		frame++
		timer.Reset(profile.Interval)
	}
}

func reactionForEmotion(emotion string) deviceReaction {
	switch emotion {
	case "surprised":
		return deviceReaction{Yaw: 0, Pitch: 18, Red: 168, Green: 70, Blue: 150, LEDPattern: "sparkle", LEDSpeed: 70 * time.Millisecond}
	case "happy", "laughing":
		return deviceReaction{Yaw: 12, Pitch: 14, Red: 168, Green: 40, Blue: 120, LEDPattern: "sparkle", LEDSpeed: 120 * time.Millisecond}
	case "angry":
		return deviceReaction{Yaw: -12, Pitch: 4, Red: 168, Green: 0, Blue: 0, LEDPattern: "fast_pulse", LEDSpeed: 80 * time.Millisecond}
	case "sad", "crying":
		return deviceReaction{Yaw: 0, Pitch: 0, Red: 20, Green: 30, Blue: 168, LEDPattern: "slow_breathe", LEDSpeed: 360 * time.Millisecond}
	case "sleepy":
		return deviceReaction{Yaw: 0, Pitch: 3, Red: 40, Green: 20, Blue: 80, LEDPattern: "slow_breathe", LEDSpeed: 520 * time.Millisecond}
	case "doubtful":
		return deviceReaction{Yaw: -10, Pitch: 12, Red: 120, Green: 40, Blue: 168, LEDPattern: "curious", LEDSpeed: 180 * time.Millisecond}
	default:
		return deviceReaction{Yaw: 0, Pitch: 8, Red: 40, Green: 20, Blue: 80, LEDPattern: "soft_glow", LEDSpeed: 260 * time.Millisecond}
	}
}

func (s *wsSession) reactToEmotion(ctx context.Context, emotion string) {
	// Stock StackChan firmware can become unresponsive when a burst of MCP
	// head/LED commands overlaps audio playback. Keep the normal protocol-level
	// facial emotion, but make every extra physical speech reaction opt-in as a
	// single safety switch.
	if !aiBool(ctx, "stackchan_speech_motion_enabled", false) {
		g.Log().Infof(ctx, "[REACTION] device=%s emotion=%s physical_reaction=false", s.deviceID, emotion)
		return
	}
	device := s.deviceMCP
	if device == nil {
		return
	}
	reaction := reactionForEmotion(emotion)
	headMotion := device.hasTool("self.robot.set_head_angles")
	ledMotion := device.hasTool("self.robot.set_led_color")
	if aiBool(ctx, "stackchan_emotion_led_enabled", false) {
		ledMotion = false
	}
	go func() {
		s.actionMu.Lock()
		defer s.actionMu.Unlock()
		if headMotion {
			s.animateHeadGesture(ctx, device, emotion, reaction)
		}
		if ledMotion {
			_ = s.animateLED(ctx, device, emotion, reaction)
		}
	}()
	g.Log().Infof(ctx, "[REACTION] device=%s emotion=%s speech_motion=%t", s.deviceID, emotion, headMotion)
}

func loudSoundDetected(pcm []int16) bool {
	if len(pcm) == 0 {
		return false
	}
	var energy int64
	peak := 0
	for _, sample := range pcm {
		value := int(sample)
		if value < 0 {
			value = -value
		}
		if value > peak {
			peak = value
		}
		energy += int64(sample) * int64(sample)
	}
	return peak >= loudStartlePeakThreshold || energy/int64(len(pcm)) >= loudStartleRMSThreshold
}

func loudStartleCooldown(ctx context.Context) time.Duration {
	return time.Duration(min(60, max(3, aiInt(ctx, "stackchan_loud_startle_cooldown_seconds", 8)))) * time.Second
}

func (s *wsSession) maybeReactToLoudSound(ctx context.Context, pcm []int16) {
	if !aiBool(ctx, "stackchan_loud_startle_enabled", false) || !loudSoundDetected(pcm) {
		return
	}
	device := s.deviceMCP
	if device == nil {
		return
	}
	now := time.Now()
	s.mu.Lock()
	if s.loudStartleBusy || (!s.lastLoudStartle.IsZero() && now.Sub(s.lastLoudStartle) < loudStartleCooldown(ctx)) {
		s.mu.Unlock()
		return
	}
	s.loudStartleBusy = true
	s.lastLoudStartle = now
	s.mu.Unlock()
	go s.runLoudStartle(ctx, device)
}

func (s *wsSession) runLoudStartle(ctx context.Context, device *deviceMCPClient) {
	defer func() {
		s.mu.Lock()
		s.loudStartleBusy = false
		s.mu.Unlock()
	}()
	s.actionMu.Lock()
	defer s.actionMu.Unlock()
	reaction := reactionForEmotion("surprised")
	if device.hasTool("self.robot.set_head_angles") {
		steps := [][3]int{{-16, 18, 360}, {14, 4, 360}, {0, 10, 260}, {0, 8, 180}}
		if _, err := device.callHeadSequence(ctx, steps); err != nil {
			g.Log().Warningf(ctx, "[STARTLE] device=%s head: %v", s.deviceID, err)
			return
		}
	}
	if device.hasTool("self.robot.set_led_color") {
		if err := s.animateLED(ctx, device, "startle", reaction); err != nil {
			g.Log().Warningf(ctx, "[STARTLE] device=%s led: %v", s.deviceID, err)
			return
		}
	}
	g.Log().Infof(ctx, "[STARTLE] device=%s loud sound reaction complete", s.deviceID)
}

func (s *wsSession) animateHeadGesture(ctx context.Context, device *deviceMCPClient, emotion string, reaction deviceReaction) {
	steps := headGestureSteps(emotion, reaction, communityMotionSpeed(ctx, 180))
	if _, err := device.callHeadSequence(ctx, steps); err != nil {
		g.Log().Warningf(ctx, "[REACTION] device=%s head emotion=%s: %v", s.deviceID, emotion, err)
		return
	}
	g.Log().Infof(ctx, "[REACTION] device=%s head emotion=%s steps=%d", s.deviceID, emotion, len(steps))
}

// headGestureSteps gives speech a short feature-animation-style anticipation,
// emotional accent, and settle. The deterministic sequence keeps character
// motion expressive without letting the language model drive raw servos.
func headGestureSteps(emotion string, reaction deviceReaction, speed int) [][3]int {
	speed = min(340, max(120, speed))
	accentSpeed := min(400, speed+55)
	settleSpeed := max(110, speed-25)
	settle := [3]int{0, 8, settleSpeed}
	switch emotion {
	case "surprised":
		return [][3]int{{-12, 6, accentSpeed}, {12, 18, min(400, accentSpeed+35)}, {-8, 16, accentSpeed}, {6, 11, speed}, settle}
	case "happy", "laughing":
		return [][3]int{{-7, 7, speed}, {reaction.Yaw, reaction.Pitch, accentSpeed}, {-5, 12, speed}, settle}
	case "angry":
		return [][3]int{{8, 10, speed}, {reaction.Yaw, reaction.Pitch, accentSpeed}, {8, 5, speed}, settle}
	case "sad", "crying":
		return [][3]int{{-5, 9, settleSpeed}, {4, 3, speed}, {0, 1, settleSpeed}, settle}
	case "sleepy":
		return [][3]int{{-4, 6, settleSpeed}, {4, 3, settleSpeed}, settle}
	case "doubtful":
		return [][3]int{{7, 7, speed}, {reaction.Yaw, reaction.Pitch, accentSpeed}, {8, 15, speed}, settle}
	default:
		return [][3]int{{-4, 7, speed}, {reaction.Yaw + 6, reaction.Pitch + 4, accentSpeed}, settle}
	}
}

func (s *wsSession) animateLED(ctx context.Context, device *deviceMCPClient, emotion string, reaction deviceReaction) error {
	generation := atomic.AddInt64(&s.ledGeneration, 1)
	steps := ledPatternSteps(reaction)
	for _, step := range steps {
		if atomic.LoadInt64(&s.ledGeneration) != generation {
			return nil
		}
		if _, err := device.callTool(ctx, "self.robot.set_led_color", map[string]any{"red": step.Red, "green": step.Green, "blue": step.Blue}); err != nil {
			g.Log().Warningf(ctx, "[REACTION] device=%s led emotion=%s: %v", s.deviceID, emotion, err)
			return err
		}
		timer := time.NewTimer(reaction.LEDSpeed)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	g.Log().Infof(ctx, "[REACTION] device=%s led emotion=%s pattern=%s steps=%d", s.deviceID, emotion, reaction.LEDPattern, len(steps))
	return nil
}

func ledPatternSteps(reaction deviceReaction) []deviceReaction {
	scale := func(percent int) deviceReaction {
		return deviceReaction{
			Red:   reaction.Red * percent / 100,
			Green: reaction.Green * percent / 100,
			Blue:  reaction.Blue * percent / 100,
		}
	}
	switch reaction.LEDPattern {
	case "fast_pulse":
		return []deviceReaction{scale(100), scale(20), scale(100), scale(15), scale(80)}
	case "slow_breathe":
		return []deviceReaction{scale(15), scale(35), scale(70), scale(100), scale(70), scale(35)}
	case "sparkle":
		return []deviceReaction{
			scale(75),
			{Red: 168, Green: 80, Blue: 150},
			scale(35),
			{Red: 120, Green: 70, Blue: 168},
			scale(90),
		}
	case "curious":
		return []deviceReaction{scale(50), scale(100), {Red: 40, Green: 20, Blue: 168}, scale(80)}
	default:
		return []deviceReaction{scale(45), scale(65), scale(80), scale(65)}
	}
}

func autonomousActionInterval(ctx context.Context) time.Duration {
	return time.Duration(min(300, max(5, aiInt(ctx, "autonomous_action_interval_seconds", 18)))) * time.Second
}

func autonomousReaction(sequence int64) deviceReaction {
	switch sequence % 4 {
	case 1:
		return deviceReaction{Yaw: -6, Pitch: 10, Red: 70, Green: 24, Blue: 140, LEDPattern: "curious", LEDSpeed: 170 * time.Millisecond}
	case 2:
		return deviceReaction{Yaw: 7, Pitch: 13, Red: 130, Green: 36, Blue: 115, LEDPattern: "sparkle", LEDSpeed: 135 * time.Millisecond}
	case 3:
		return deviceReaction{Yaw: 0, Pitch: 6, Red: 28, Green: 18, Blue: 92, LEDPattern: "slow_breathe", LEDSpeed: 300 * time.Millisecond}
	default:
		return deviceReaction{Yaw: 4, Pitch: 9, Red: 45, Green: 20, Blue: 120, LEDPattern: "soft_glow", LEDSpeed: 220 * time.Millisecond}
	}
}

func (s *wsSession) canStartAutonomousAction() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.autonomousBusy || s.loudStartleBusy || s.isListening || s.playbackStarted || s.opusEnc != nil || len(s.frameQueue) > 0 || (s.activity != nil && s.activity.responseBusy()) {
		return false
	}
	s.autonomousBusy = true
	return true
}

func (s *wsSession) finishAutonomousAction() {
	s.mu.Lock()
	s.autonomousBusy = false
	s.mu.Unlock()
}

func (s *wsSession) autonomousLoop(ctx context.Context) {
	if !aiBool(ctx, "autonomous_actions_enabled", false) {
		return
	}
	timer := time.NewTimer(autonomousActionInterval(ctx))
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
			if s.canStartAutonomousAction() {
				go s.runAutonomousAction(ctx)
			}
			timer.Reset(autonomousActionInterval(ctx))
		}
	}
}

func (s *wsSession) runAutonomousAction(ctx context.Context) {
	defer s.finishAutonomousAction()
	s.actionMu.Lock()
	defer s.actionMu.Unlock()
	device := s.deviceMCP
	if device == nil {
		return
	}
	sequence := atomic.AddInt64(&s.autonomousCount, 1)
	reaction := autonomousReaction(sequence)
	hasLED := device.hasTool("self.robot.set_led_color")
	hasHead := device.hasTool("self.robot.set_head_angles")
	if !hasLED && !hasHead {
		return
	}
	if hasHead {
		s.animateHeadGesture(ctx, device, "heartbeat", reaction)
	}
	if hasLED {
		s.animateLED(ctx, device, "heartbeat", reaction)
	}
	g.Log().Infof(ctx, "[AUTONOMY] device=%s heartbeat action=%d led=%t head=%t", s.deviceID, sequence, hasLED, hasHead)
}

func (s *wsSession) alignFaceIfDue(ctx context.Context, trigger string) {
	if !aiBool(ctx, "face_contact_enabled", false) {
		return
	}
	device := s.deviceMCP
	if device == nil || !device.hasTool("self.camera.take_photo") || !device.hasTool("self.robot.set_head_angles") {
		return
	}
	interval := time.Duration(max(10, aiInt(ctx, "face_contact_interval_seconds", 45))) * time.Second
	now := time.Now()
	s.mu.Lock()
	if s.faceContactBusy || (!s.lastFaceContact.IsZero() && now.Sub(s.lastFaceContact) < interval) {
		s.mu.Unlock()
		return
	}
	s.faceContactBusy = true
	s.lastFaceContact = now
	s.mu.Unlock()

	go func() {
		defer func() {
			s.mu.Lock()
			s.faceContactBusy = false
			s.mu.Unlock()
		}()
		s.alignFace(ctx, trigger, device)
	}()
}

func (s *wsSession) alignFace(ctx context.Context, trigger string, device *deviceMCPClient) {
	question := `For Eve's eye contact, find the largest human face in this camera image. Return only JSON: {"face":true|false,"horizontal":"left|center|right","vertical":"up|center|down","confidence":0.0}. Use "center" when already near the center.`
	result, err := device.callTool(ctx, "self.camera.take_photo", map[string]any{"question": question})
	if err != nil {
		g.Log().Warningf(ctx, "[FACE] device=%s trigger=%s camera: %v", s.deviceID, trigger, err)
		return
	}
	obs, ok := parseFaceObservation(result)
	if !ok {
		g.Log().Warningf(ctx, "[FACE] device=%s trigger=%s parse failed result=%s", s.deviceID, trigger, result)
		return
	}
	if !obs.Face || obs.Confidence < 0.35 {
		g.Log().Infof(ctx, "[FACE] device=%s trigger=%s no confident face observation=%+v", s.deviceID, trigger, obs)
		return
	}
	yaw, pitch := faceContactAngles(obs)
	if _, err := device.callTool(ctx, "self.robot.set_head_angles", map[string]any{"yaw": yaw, "pitch": pitch, "speed": 140}); err != nil {
		g.Log().Warningf(ctx, "[FACE] device=%s trigger=%s head: %v", s.deviceID, trigger, err)
		return
	}
	g.Log().Infof(ctx, "[FACE] device=%s trigger=%s horizontal=%s vertical=%s confidence=%.2f yaw=%d pitch=%d", s.deviceID, trigger, obs.Horizontal, obs.Vertical, obs.Confidence, yaw, pitch)
}

func parseFaceObservation(text string) (faceObservation, bool) {
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start < 0 || end <= start {
		return faceObservation{}, false
	}
	var obs faceObservation
	if err := json.Unmarshal([]byte(text[start:end+1]), &obs); err != nil {
		return faceObservation{}, false
	}
	obs.Horizontal = strings.ToLower(strings.TrimSpace(obs.Horizontal))
	obs.Vertical = strings.ToLower(strings.TrimSpace(obs.Vertical))
	if obs.Horizontal == "" {
		obs.Horizontal = "center"
	}
	if obs.Vertical == "" {
		obs.Vertical = "center"
	}
	return obs, true
}

func faceContactAngles(obs faceObservation) (int, int) {
	yaw := 0
	switch obs.Horizontal {
	case "left":
		yaw = -10
	case "right":
		yaw = 10
	}
	pitch := 8
	switch obs.Vertical {
	case "up":
		pitch = 14
	case "down":
		pitch = 3
	}
	return yaw, pitch
}

func (s *wsSession) handleListen(ctx context.Context, msg map[string]any) {
	state, _ := msg["state"].(string)
	switch state {

	case "detect":
		s.activity.wake(time.Now())
		// Wake word: cancel in-progress response and unblock device VAD.
		_ = s.rt.CancelResponse()
		s.mu.Lock()
		s.opusEnc = nil
		s.playbackStarted = false
		if s.prebufferTimer != nil {
			s.prebufferTimer.Stop()
			s.prebufferTimer = nil
		}
		s.mu.Unlock()
		s.drainFrameQueue()
		if aware, ok := s.rt.(PlaybackStateAware); ok {
			aware.InterruptPlayback()
		}
		_ = s.sendJSON(map[string]any{"type": "tts", "state": "stop"})

	case "start":
		if mode, _ := msg["mode"].(string); mode == "manual" {
			s.activity.wake(time.Now())
		}
		if s.activity.expired(time.Now()) {
			s.conn.Close()
			return
		}
		// New utterance: reset the OPUS decoder for a clean stream.
		dec, err := newOpusDecoder()
		if err != nil {
			g.Log().Warningf(ctx, "[WS] device=%s decoder reset: %v", s.deviceID, err)
		} else {
			s.opusDec = dec
		}
		s.mu.Lock()
		s.isListening = true
		s.lastInteraction = time.Now()
		s.vadHeardSpeech = false
		s.vadSilenceSamples = 0
		s.inputFrames = 0
		s.inputSamples = 0
		s.mu.Unlock()
		atomic.StoreInt32(&s.inputAudioLogged, 0)
		atomic.StoreInt32(&s.inputDecodeErrors, 0)
		g.Log().Infof(ctx, "[WS] device=%s listening started", s.deviceID)

	case "stop":
		s.commitListening(ctx, "device")
	}
}

// observeServerVAD supplies end-of-utterance detection for HTTP STT pipelines.
// Realtime providers keep using their native server VAD and never enter here.
func (s *wsSession) observeServerVAD(ctx context.Context, pcm []int16) bool {
	if !s.serverVAD || len(pcm) == 0 {
		return false
	}
	var energy int64
	for _, sample := range pcm {
		energy += int64(sample) * int64(sample)
	}

	s.mu.Lock()
	if !s.isListening {
		s.mu.Unlock()
		return false
	}
	if energy/int64(len(pcm)) > serverVADThreshold {
		s.vadHeardSpeech = true
		s.vadSilenceSamples = 0
		s.mu.Unlock()
		return false
	}
	if !s.vadHeardSpeech {
		s.mu.Unlock()
		return false
	}
	s.vadSilenceSamples += len(pcm)
	shouldCommit := time.Duration(s.vadSilenceSamples)*time.Second/serverSampleRate >= serverVADSilenceTime
	s.mu.Unlock()
	if shouldCommit {
		return s.commitListening(ctx, "server_vad")
	}
	return false
}

func (s *wsSession) commitListening(ctx context.Context, source string) bool {
	s.mu.Lock()
	wasListening := s.isListening
	s.isListening = false
	s.vadHeardSpeech = false
	s.vadSilenceSamples = 0
	inputFrames := s.inputFrames
	inputSamples := s.inputSamples
	s.mu.Unlock()
	if !wasListening || !s.activity.commit(time.Now()) {
		return false
	}
	s.mu.Lock()
	s.listenStopped = time.Now()
	s.mu.Unlock()
	atomic.StoreInt32(&s.firstAudioLogged, 0)
	if err := s.rt.CommitAudio(); err != nil {
		g.Log().Warningf(ctx, "[WS] device=%s audio commit failed source=%s: %v", s.deviceID, source, err)
		return false
	}
	g.Log().Infof(ctx, "[WS] device=%s listening stopped source=%s frames=%d samples=%d duration_ms=%d (committed)", s.deviceID, source, inputFrames, inputSamples, inputSamples*1000/serverSampleRate)
	return true
}

// logTurnLatency emits cumulative timings from the device's listen:stop event.
// It is intentionally provider-neutral so logs can compare realtime and HTTP
// pipelines without exposing audio or credentials.
func (s *wsSession) logTurnLatency(ctx context.Context, stage string) {
	s.mu.Lock()
	stopped := s.listenStopped
	s.mu.Unlock()
	if stopped.IsZero() {
		return
	}
	g.Log().Infof(ctx, "[LAT] device=%s stage=%s since_listen_stop_ms=%d", s.deviceID, stage, time.Since(stopped).Milliseconds())
}

func (s *wsSession) sendJSON(v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.WriteMessage(websocket.TextMessage, data)
}

func (s *wsSession) sendAudio(frame []byte) error {
	// BinaryProtocol3: [0x00][0x00][payload_size hi][payload_size lo][payload]
	header := [4]byte{0x00, 0x00, byte(len(frame) >> 8), byte(len(frame))}
	msg := append(header[:], frame...)
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.WriteMessage(websocket.BinaryMessage, msg)
}

func (s *wsSession) pingLoop(ctx context.Context) {
	ticker := time.NewTicker(50 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			s.writeMu.Lock()
			_ = s.conn.WriteControl(websocket.PingMessage, nil, time.Now().Add(5*time.Second))
			s.writeMu.Unlock()
		case <-ctx.Done():
			return
		}
	}
}
