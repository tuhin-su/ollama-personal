package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"

	"github.com/gin-gonic/gin"
)

// TelemetryHub manages real-time broadcast of inference events to listeners and visualizers.
type TelemetryHub struct {
	mu          sync.RWMutex
	subscribers atomic.Int32
	clients     map[chan string]struct{}
}

// GlobalTelemetry is the singleton telemetry hub for Ollama inference events.
var GlobalTelemetry = &TelemetryHub{
	clients: make(map[chan string]struct{}),
}

// Active returns true if at least one client/visualizer is actively listening.
// It uses an atomic load with zero lock contention and zero allocations.
func (h *TelemetryHub) Active() bool {
	if h == nil {
		return false
	}
	return h.subscribers.Load() > 0
}

// Broadcast sends an event to all connected telemetry listeners if active.
func (h *TelemetryHub) Broadcast(eventType string, data any) {
	if !h.Active() {
		return
	}
	h.mu.RLock()
	defer h.mu.RUnlock()

	if len(h.clients) == 0 {
		return
	}

	payload, err := json.Marshal(data)
	if err != nil {
		return
	}

	msg := fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, string(payload))
	for ch := range h.clients {
		select {
		case ch <- msg:
		default:
		}
	}
}

// DevTelemetryHandler serves real-time Server-Sent Events to visualizer listeners.
func (s *Server) DevTelemetryHandler(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Writer.WriteHeader(http.StatusOK)
	c.Writer.Flush()

	eventChan := make(chan string, 128)
	GlobalTelemetry.mu.Lock()
	GlobalTelemetry.clients[eventChan] = struct{}{}
	GlobalTelemetry.subscribers.Add(1)
	GlobalTelemetry.mu.Unlock()

	defer func() {
		GlobalTelemetry.mu.Lock()
		delete(GlobalTelemetry.clients, eventChan)
		GlobalTelemetry.subscribers.Add(-1)
		close(eventChan)
		GlobalTelemetry.mu.Unlock()
	}()

	c.Stream(func(w io.Writer) bool {
		select {
		case <-c.Request.Context().Done():
			return false
		case msg, ok := <-eventChan:
			if !ok {
				return false
			}
			_, _ = w.Write([]byte(msg))
			return true
		}
	})
}
