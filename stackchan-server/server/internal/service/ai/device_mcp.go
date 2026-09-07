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
		return tool.Description + "\nUse this whenever the user asks what Eve can see, asks about the camera, or asks where an object/finger/person is. Pass the user's visual question as `question` in Korean."
	case "self.robot.set_head_angles":
		return tool.Description + "\nUse this for explicit motion requests and small natural reactions. Keep normal reactions within yaw -20..20 and pitch 0..20 unless the user asks for a larger movement."
	case "self.robot.set_led_color":
		return tool.Description + "\nUse this as Eve's subtle emotional accent light, not as a room light."
	default:
		return tool.Description
	}
}

func (c *deviceMCPClient) openAITools() []map[string]any {
	c.mu.RLock()
	defer c.mu.RUnlock()
	tools := make([]map[string]any, 0, len(c.tools))
	for _, tool := range c.tools {
		tools = append(tools, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": openAIDeviceToolName(tool.Name), "description": enhancedDeviceToolDescription(tool), "parameters": tool.InputSchema,
			},
		})
	}
	return tools
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
