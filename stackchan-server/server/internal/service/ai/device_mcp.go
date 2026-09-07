/*
SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
SPDX-License-Identifier: MIT
*/

package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const deviceMCPTimeout = 30 * time.Second

type deviceMCPResponse struct {
	Result json.RawMessage `json:"result"`
	Error  *struct {
		Message string `json:"message"`
	} `json:"error"`
}

type deviceMCPTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type deviceMCPClient struct {
	send      func(any) error
	sessionID string
	nextID    int64
	mu        sync.RWMutex
	pending   map[int64]chan deviceMCPResponse
	tools     []deviceMCPTool
	nameMap   map[string]string
}

func newDeviceMCPClient(sessionID string, send func(any) error) *deviceMCPClient {
	return &deviceMCPClient{
		send: send, sessionID: sessionID, pending: map[int64]chan deviceMCPResponse{},
		nameMap: map[string]string{},
	}
}

func (c *deviceMCPClient) request(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := atomic.AddInt64(&c.nextID, 1)
	ch := make(chan deviceMCPResponse, 1)
	c.mu.Lock()
	c.pending[id] = ch
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
	}()

	err := c.send(map[string]any{
		"session_id": c.sessionID,
		"type":       "mcp",
		"payload": map[string]any{
			"jsonrpc": "2.0", "method": method, "params": params, "id": id,
		},
	})
	if err != nil {
		return nil, err
	}

	timer := time.NewTimer(deviceMCPTimeout)
	defer timer.Stop()
	select {
	case resp := <-ch:
		if resp.Error != nil {
			return nil, fmt.Errorf("device MCP %s: %s", method, resp.Error.Message)
		}
		return resp.Result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-timer.C:
		return nil, fmt.Errorf("device MCP %s timed out", method)
	}
}

func (c *deviceMCPClient) handle(payload any) bool {
	b, err := json.Marshal(payload)
	if err != nil {
		return false
	}
	var envelope struct {
		ID int64 `json:"id"`
		deviceMCPResponse
	}
	if json.Unmarshal(b, &envelope) != nil || envelope.ID == 0 {
		return false
	}
	c.mu.RLock()
	ch := c.pending[envelope.ID]
	c.mu.RUnlock()
	if ch == nil {
		return false
	}
	select {
	case ch <- envelope.deviceMCPResponse:
	default:
	}
	return true
}

func (c *deviceMCPClient) initialize(ctx context.Context, visionURL, visionToken string) error {
	capabilities := map[string]any{}
	if visionURL != "" {
		capabilities["vision"] = map[string]any{"url": visionURL, "token": visionToken}
	}
	if _, err := c.request(ctx, "initialize", map[string]any{"capabilities": capabilities}); err != nil {
		return err
	}

	var tools []deviceMCPTool
	cursor := ""
	for {
		result, err := c.request(ctx, "tools/list", map[string]any{"cursor": cursor, "withUserTools": false})
		if err != nil {
			return err
		}
		var page struct {
			Tools      []deviceMCPTool `json:"tools"`
			NextCursor string          `json:"nextCursor"`
		}
		if err := json.Unmarshal(result, &page); err != nil {
			return fmt.Errorf("device MCP tools/list response: %w", err)
		}
		tools = append(tools, page.Tools...)
		if page.NextCursor == "" {
			break
		}
		cursor = page.NextCursor
	}

	nameMap := make(map[string]string, len(tools))
	for _, tool := range tools {
		nameMap[openAIDeviceToolName(tool.Name)] = tool.Name
	}
	c.mu.Lock()
	c.tools = tools
	c.nameMap = nameMap
	c.mu.Unlock()
	return nil
}

func (c *deviceMCPClient) toolNames() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	names := make([]string, 0, len(c.tools))
	for _, tool := range c.tools {
		names = append(names, tool.Name)
	}
	return names
}

func (c *deviceMCPClient) hasTool(deviceName string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, original := range c.nameMap {
		if original == deviceName {
			return true
		}
	}
	return false
}

func openAIDeviceToolName(name string) string {
	return "device__" + strings.NewReplacer(".", "__", "-", "_").Replace(name)
}

func enhancedDeviceToolDescription(tool deviceMCPTool) string {
	switch tool.Name {
	case "self.camera.take_photo":
		return tool.Description + "\nUse only for explicit visual requests where the user clearly asks Eve to use the camera, take a photo, or look at what is in front of her. Pass the user's visual question as `question` in Korean."
	case "self.robot.set_head_angles":
		return tool.Description + "\nUse only for explicit motion requests. Keep yaw in -20..20, pitch in 5..20, and speed at least 100 unless the device reports a different safe range."
	case "self.robot.set_led_color":
		return tool.Description + "\nUse this as Eve's subtle emotional accent light, not as a room light."
	default:
		return tool.Description
	}
}

func (c *deviceMCPClient) openAITools() []map[string]any {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.communityOpenAIToolsLocked()
}

func (c *deviceMCPClient) callTool(ctx context.Context, deviceName string, arguments map[string]any) (string, error) {
	result, err := c.request(ctx, "tools/call", map[string]any{"name": deviceName, "arguments": arguments})
	if err != nil {
		return "", err
	}
	var decoded struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if json.Unmarshal(result, &decoded) == nil && len(decoded.Content) > 0 {
		parts := make([]string, 0, len(decoded.Content))
		for _, item := range decoded.Content {
			if item.Text != "" {
				parts = append(parts, item.Text)
			}
		}
		text := strings.Join(parts, "\n")
		if decoded.IsError {
			return "", fmt.Errorf("device tool failed: %s", text)
		}
		return text, nil
	}
	return string(result), nil
}

func (c *deviceMCPClient) call(ctx context.Context, openAIName string, arguments map[string]any) (string, error) {
	if strings.HasPrefix(openAIName, "stackchan_") {
		return c.callCommunityTool(ctx, openAIName, arguments)
	}
	c.mu.RLock()
	deviceName := c.nameMap[openAIName]
	c.mu.RUnlock()
	if deviceName == "" {
		return "", fmt.Errorf("unknown device tool %q", openAIName)
	}
	return c.callTool(ctx, deviceName, arguments)
}

type deviceToolAware interface {
	SetDeviceTools(*deviceMCPClient)
}

func (c *deviceMCPClient) communityOpenAIToolsLocked() []map[string]any {
	tools := []map[string]any{}
	if c.hasToolLocked("self.camera.take_photo") {
		tools = append(tools, communityTool("stackchan_see",
			"Use Eve's camera for an explicit visual request. Call only when the user clearly asks for camera/photo/vision/looking at the scene. The answer must be based only on the returned observation.",
			objectSchema(map[string]any{
				"question": map[string]any{"type": "string", "description": "The user's visual question in Korean."},
			}, []string{"question"})))
	}
	if c.hasToolLocked("self.robot.set_head_angles") {
		tools = append(tools,
			communityTool("stackchan_move",
				"Move Eve's head for an explicit user motion request. This is not for idle animation; the firmware keeps Eve lively by itself.",
				objectSchema(map[string]any{
					"yaw":   map[string]any{"type": "number", "minimum": -20, "maximum": 20, "description": "Left/right head angle. Negative is left, positive is right."},
					"pitch": map[string]any{"type": "number", "minimum": 5, "maximum": 20, "description": "Up/down head angle in the safe normal range."},
					"speed": map[string]any{"type": "number", "minimum": 100, "maximum": 400, "description": "Servo speed. Use 140-220 for normal motion."},
				}, []string{"yaw", "pitch"})),
			communityTool("stackchan_nod",
				"Make Eve nod once when the user explicitly asks her to nod or agree physically.",
				objectSchema(map[string]any{}, nil)),
			communityTool("stackchan_shake",
				"Make Eve shake her head once when the user explicitly asks her to shake her head or disagree physically.",
				objectSchema(map[string]any{}, nil)),
		)
	}
	if c.hasAnyToolLocked([]string{
		"self.avatar.set_expression", "self.avatar.set_face", "self.robot.set_expression",
		"self.robot.set_avatar", "self.display.set_avatar", "self.display.set_expression",
		"self.face.set_expression", "self.robot.set_led_color",
	}) {
		tools = append(tools, communityTool("stackchan_face",
			"Set Eve's high-level expression. Prefer this over low-level LED or servo tools. Expressions follow common Stack-chan avatar-style moods.",
			objectSchema(map[string]any{
				"expression": map[string]any{"type": "string", "enum": []string{"calm", "happy", "thinking", "shy", "pouty", "surprised", "sad", "sleepy"}, "description": "Eve's expression."},
			}, []string{"expression"})))
	}
	tools = append(tools, communityTool("stackchan_status",
		"Report Eve's currently available high-level body capabilities. Use only when the user asks what the device can do or asks for device status.",
		objectSchema(map[string]any{}, nil)))
	tools = append(tools, communityTool("stackchan_health",
		"Check Eve's local device bridge health and available body tools. Use only for troubleshooting.",
		objectSchema(map[string]any{}, nil)))
	if c.hasAnyToolLocked([]string{"self.sensor.get_environment", "self.sensors.get_environment", "self.env.get", "self.robot.get_sensors"}) {
		tools = append(tools, communityTool("stackchan_sense",
			"Read Eve's environmental sensors when available.",
			objectSchema(map[string]any{}, nil)))
	}
	return tools
}

func communityTool(name, description string, parameters map[string]any) map[string]any {
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        name,
			"description": description,
			"parameters":  parameters,
		},
	}
}

func objectSchema(properties map[string]any, required []string) map[string]any {
	schema := map[string]any{"type": "object", "properties": properties, "additionalProperties": false}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

func (c *deviceMCPClient) hasToolLocked(deviceName string) bool {
	for _, original := range c.nameMap {
		if original == deviceName {
			return true
		}
	}
	return false
}

func (c *deviceMCPClient) hasAnyToolLocked(deviceNames []string) bool {
	for _, name := range deviceNames {
		if c.hasToolLocked(name) {
			return true
		}
	}
	return false
}

func (c *deviceMCPClient) callCommunityTool(ctx context.Context, name string, arguments map[string]any) (string, error) {
	switch name {
	case "stackchan_see":
		question := strings.TrimSpace(stringArg(arguments, "question"))
		if question == "" {
			question = "교수님이 명시적으로 카메라 확인을 요청했다. 화면에 보이는 것을 짧고 정확하게 설명해줘."
		}
		return c.callTool(ctx, "self.camera.take_photo", map[string]any{"question": question})
	case "stackchan_move":
		yaw := int(clampFloat(numberArg(arguments, "yaw", 0), -20, 20))
		pitch := int(clampFloat(numberArg(arguments, "pitch", 8), 5, 20))
		speed := int(clampFloat(numberArg(arguments, "speed", float64(communityMotionSpeed(ctx, 160))), 100, 400))
		return c.callHead(ctx, yaw, pitch, speed)
	case "stackchan_nod":
		speed := communityMotionSpeed(ctx, 180)
		return c.callHeadSequence(ctx, [][3]int{{0, 16, speed}, {0, 6, speed}, {0, 14, speed}, {0, 8, communityMotionSpeed(ctx, 160)}})
	case "stackchan_shake":
		speed := communityMotionSpeed(ctx, 190)
		return c.callHeadSequence(ctx, [][3]int{{-14, 8, speed}, {14, 8, speed}, {-10, 8, communityMotionSpeed(ctx, 180)}, {0, 8, communityMotionSpeed(ctx, 160)}})
	case "stackchan_face":
		expression := normalizeCommunityExpression(stringArg(arguments, "expression"))
		return c.callExpression(ctx, expression)
	case "stackchan_status":
		return c.communityStatusJSON("status"), nil
	case "stackchan_health":
		return c.communityStatusJSON("healthy"), nil
	case "stackchan_sense":
		return c.callFirstAvailable(ctx, []string{"self.sensor.get_environment", "self.sensors.get_environment", "self.env.get", "self.robot.get_sensors"}, nil)
	default:
		return "", fmt.Errorf("unknown StackChan community tool %q", name)
	}
}

func (c *deviceMCPClient) callHead(ctx context.Context, yaw, pitch, speed int) (string, error) {
	if !c.hasTool("self.robot.set_head_angles") {
		return "", fmt.Errorf("head movement tool is unavailable")
	}
	return c.callTool(ctx, "self.robot.set_head_angles", map[string]any{"yaw": yaw, "pitch": pitch, "speed": speed})
}

func (c *deviceMCPClient) callHeadSequence(ctx context.Context, steps [][3]int) (string, error) {
	for _, step := range steps {
		if _, err := c.callHead(ctx, step[0], step[1], step[2]); err != nil {
			return "", err
		}
		timer := time.NewTimer(communityMotionStepDelay(ctx))
		select {
		case <-ctx.Done():
			timer.Stop()
			return "", ctx.Err()
		case <-timer.C:
		}
	}
	return "ok", nil
}

func (c *deviceMCPClient) callExpression(ctx context.Context, expression string) (string, error) {
	args := map[string]any{"expression": expression, "name": expression, "face": expression, "emotion": expression}
	if result, err := c.callFirstAvailable(ctx, []string{
		"self.avatar.set_expression", "self.avatar.set_face", "self.robot.set_expression",
		"self.robot.set_avatar", "self.display.set_avatar", "self.display.set_expression",
		"self.face.set_expression",
	}, args); err == nil {
		return result, nil
	}
	if c.hasTool("self.robot.set_led_color") {
		color := communityExpressionColor(ctx, expression)
		return c.callTool(ctx, "self.robot.set_led_color", map[string]any{"red": color[0], "green": color[1], "blue": color[2]})
	}
	return "", fmt.Errorf("expression tool is unavailable")
}

func (c *deviceMCPClient) callFirstAvailable(ctx context.Context, names []string, args map[string]any) (string, error) {
	for _, name := range names {
		if c.hasTool(name) {
			if args == nil {
				args = map[string]any{}
			}
			return c.callTool(ctx, name, args)
		}
	}
	return "", fmt.Errorf("no matching device tool is available")
}

func (c *deviceMCPClient) communityStatusJSON(status string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	capabilities := map[string]bool{
		"see":    c.hasToolLocked("self.camera.take_photo"),
		"move":   c.hasToolLocked("self.robot.set_head_angles"),
		"face":   c.hasAnyToolLocked([]string{"self.avatar.set_expression", "self.avatar.set_face", "self.robot.set_expression", "self.robot.set_avatar", "self.display.set_avatar", "self.display.set_expression", "self.face.set_expression", "self.robot.set_led_color"}),
		"sense":  c.hasAnyToolLocked([]string{"self.sensor.get_environment", "self.sensors.get_environment", "self.env.get", "self.robot.get_sensors"}),
		"health": true,
		"status": true,
	}
	body := map[string]any{"status": status, "capabilities": capabilities, "raw_tools": c.toolNamesLocked()}
	b, _ := json.Marshal(body)
	return string(b)
}

func (c *deviceMCPClient) toolNamesLocked() []string {
	names := make([]string, 0, len(c.tools))
	for _, tool := range c.tools {
		names = append(names, tool.Name)
	}
	return names
}

func normalizeCommunityExpression(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "happy", "joy", "laughing", "excited":
		return "happy"
	case "thinking", "curious", "doubtful":
		return "thinking"
	case "shy", "embarrassed":
		return "shy"
	case "pouty", "angry", "sulky":
		return "pouty"
	case "surprised", "surprise":
		return "surprised"
	case "sad", "crying":
		return "sad"
	case "sleepy":
		return "sleepy"
	default:
		return "calm"
	}
}

func communityMotionSpeed(ctx context.Context, fallback int) int {
	return int(clampFloat(float64(aiInt(ctx, "stackchan_motion_speed", fallback)), 100, 400))
}

func communityMotionStepDelay(ctx context.Context) time.Duration {
	return time.Duration(clampFloat(float64(aiInt(ctx, "stackchan_motion_step_delay_ms", 220)), 50, 1000)) * time.Millisecond
}

func communityExpressionColor(ctx context.Context, expression string) [3]int {
	defaultColor := defaultCommunityExpressionColor(expression)
	raw := strings.TrimSpace(aiString(ctx, "stackchan_expression_colors", ""))
	if raw == "" {
		return defaultColor
	}
	colors := map[string][]int{}
	if json.Unmarshal([]byte(raw), &colors) != nil {
		return defaultColor
	}
	color, ok := colors[expression]
	if !ok || len(color) != 3 {
		return defaultColor
	}
	return [3]int{
		int(clampFloat(float64(color[0]), 0, 168)),
		int(clampFloat(float64(color[1]), 0, 168)),
		int(clampFloat(float64(color[2]), 0, 168)),
	}
}

func defaultCommunityExpressionColor(expression string) [3]int {
	switch expression {
	case "happy":
		return [3]int{168, 60, 130}
	case "thinking":
		return [3]int{80, 50, 168}
	case "shy":
		return [3]int{168, 70, 120}
	case "pouty":
		return [3]int{168, 20, 45}
	case "surprised":
		return [3]int{120, 90, 168}
	case "sad":
		return [3]int{20, 40, 168}
	case "sleepy":
		return [3]int{35, 20, 90}
	default:
		return [3]int{45, 24, 120}
	}
}

func stringArg(args map[string]any, key string) string {
	if args == nil {
		return ""
	}
	value, _ := args[key].(string)
	return value
}

func numberArg(args map[string]any, key string, fallback float64) float64 {
	if args == nil {
		return fallback
	}
	switch value := args[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	case json.Number:
		n, err := value.Float64()
		if err == nil {
			return n
		}
	}
	return fallback
}

func clampFloat(value, low, high float64) float64 {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}
