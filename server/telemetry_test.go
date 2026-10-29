package server

import (
	"bufio"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func TestDevTelemetryHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{}
	r.GET("/api/dev/telemetry", s.DevTelemetryHandler)

	ts := httptest.NewServer(r)
	defer ts.Close()

	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(ts.URL + "/api/dev/telemetry")
	if err != nil {
		t.Fatalf("failed to connect to /api/dev/telemetry: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got: %d", resp.StatusCode)
	}

	time.Sleep(50 * time.Millisecond)

	// Broadcast test events
	GlobalTelemetry.Broadcast("inference_start", map[string]any{"model": "test-model"})
	GlobalTelemetry.Broadcast("token", map[string]any{"model": "test-model", "token": "Hello"})

	scanner := bufio.NewScanner(resp.Body)
	var eventsReceived []string

	readDone := make(chan bool)
	go func() {
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "event: ") {
				eventsReceived = append(eventsReceived, strings.TrimPrefix(line, "event: "))
				if len(eventsReceived) >= 2 {
					readDone <- true
					return
				}
			}
		}
	}()

	select {
	case <-readDone:
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for telemetry events; received: %v", eventsReceived)
	}

	if eventsReceived[0] != "inference_start" || eventsReceived[1] != "token" {
		t.Errorf("unexpected events: %v", eventsReceived)
	}
}
