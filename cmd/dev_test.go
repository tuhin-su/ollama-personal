package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestNewDevCommand(t *testing.T) {
	cmd := NewDevCommand()
	if cmd.Use != "dev" {
		t.Fatalf("expected Use to be 'dev', got %s", cmd.Use)
	}

	infoSubCmd, _, err := cmd.Find([]string{"info"})
	if err != nil || infoSubCmd == nil {
		t.Fatalf("expected 'info' subcommand to exist under 'dev'")
	}

	for _, flagName := range []string{"all", "layers", "tensors", "parameters", "metadata", "storage", "json"} {
		if flag := infoSubCmd.Flags().Lookup(flagName); flag == nil {
			t.Errorf("expected flag --%s on 'ollama dev info'", flagName)
		}
	}
}

func TestComputeDimensions(t *testing.T) {
	modelInfo := map[string]any{
		"general.architecture":                   "qwen2",
		"qwen2.context_length":                   float64(131072),
		"qwen2.embedding_length":                 float64(3584),
		"qwen2.block_count":                      uint32(28),
		"qwen2.attention.head_count":             uint32(28),
		"qwen2.attention.head_count_kv":          uint32(4),
		"qwen2.feed_forward_length":              uint64(18944),
		"qwen2.rope.freq_base":                   float64(10000),
		"qwen2.attention.layer_norm_rms_epsilon": float64(1e-6),
	}

	tensors := []DevTensor{
		{
			Name:       "token_embd.weight",
			Shape:      []uint64{3584, 151936},
			ParamCount: 3584 * 151936,
		},
	}

	dims := computeDimensions(modelInfo, tensors)

	if dims.ContextLength != 131072 {
		t.Errorf("expected ContextLength 131072, got %d", dims.ContextLength)
	}
	if dims.EmbeddingLength != 3584 {
		t.Errorf("expected EmbeddingLength 3584, got %d", dims.EmbeddingLength)
	}
	if dims.BlockCount != 28 {
		t.Errorf("expected BlockCount 28, got %d", dims.BlockCount)
	}
	if dims.AttentionHeads != 28 {
		t.Errorf("expected AttentionHeads 28, got %d", dims.AttentionHeads)
	}
	if dims.AttentionHeadsKV != 4 {
		t.Errorf("expected AttentionHeadsKV 4, got %d", dims.AttentionHeadsKV)
	}
	if dims.GQARatio != "7:1 (GQA)" {
		t.Errorf("expected GQARatio '7:1 (GQA)', got %s", dims.GQARatio)
	}
	if dims.FeedForwardLength != 18944 {
		t.Errorf("expected FeedForwardLength 18944, got %d", dims.FeedForwardLength)
	}
	if dims.VocabSize != 151936 {
		t.Errorf("expected VocabSize 151936, got %d", dims.VocabSize)
	}
}

func TestComputeQuantDistribution(t *testing.T) {
	tensors := []DevTensor{
		{Name: "t1", Type: "Q4_K", ParamCount: 800},
		{Name: "t2", Type: "Q4_K", ParamCount: 200},
		{Name: "t3", Type: "F32", ParamCount: 100},
	}
	totalParams := uint64(1100)

	dist := computeQuantDistribution(tensors, totalParams)
	if len(dist) != 2 {
		t.Fatalf("expected 2 quant summaries, got %d", len(dist))
	}

	// Should be sorted by total params descending
	if dist[0].Type != "Q4_K" || dist[0].TensorCount != 2 || dist[0].TotalParams != 1000 {
		t.Errorf("unexpected top quant: %+v", dist[0])
	}
	if dist[1].Type != "F32" || dist[1].TensorCount != 1 || dist[1].TotalParams != 100 {
		t.Errorf("unexpected second quant: %+v", dist[1])
	}
}

func TestComputeLayerSummary(t *testing.T) {
	tensors := []DevTensor{
		{Name: "token_embd.weight", Type: "Q4_K", ParamCount: 1000},
		{Name: "blk.0.attn_q.weight", Type: "Q4_K", ParamCount: 500},
		{Name: "blk.0.attn_k.weight", Type: "Q4_K", ParamCount: 500},
		{Name: "blk.0.ffn_gate.weight", Type: "Q6_K", ParamCount: 800},
		{Name: "blk.1.attn_q.weight", Type: "Q4_K", ParamCount: 500},
		{Name: "output.weight", Type: "Q6_K", ParamCount: 1000},
	}

	layers := computeLayerSummary(tensors, 2)
	if len(layers) != 4 {
		t.Fatalf("expected 4 layer groups (Embedding, Block 0, Block 1, Output), got %d", len(layers))
	}

	if layers[0].Name != "Embedding" {
		t.Errorf("expected first layer to be Embedding, got %s", layers[0].Name)
	}
	if layers[1].Name != "Block 0" || layers[1].TensorCount != 3 {
		t.Errorf("unexpected Block 0 summary: %+v", layers[1])
	}
	if layers[2].Name != "Block 1" || layers[2].TensorCount != 1 {
		t.Errorf("unexpected Block 1 summary: %+v", layers[2])
	}
	if layers[3].Name != "Output / LM Head" {
		t.Errorf("expected last layer to be Output / LM Head, got %s", layers[3].Name)
	}
}

func TestDevInfoHandler_WithServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/api/version":
			_ = json.NewEncoder(w).Encode(map[string]string{"version": "0.0.0"})
		case r.Method == http.MethodGet && r.URL.Path == "/api/tags":
			resp := api.ListResponse{
				Models: []api.ListModelResponse{
					{Name: "test-model:latest", Model: "test-model", Digest: "sha256:aabbcc112233"},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		case r.Method == http.MethodPost && r.URL.Path == "/api/show":
			var req api.ShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			resp := api.ShowResponse{
				Details: api.ModelDetails{
					Format:            "gguf",
					Family:            "llama",
					ParameterSize:     "8B",
					QuantizationLevel: "Q4_K_M",
					Runner:            "llama",
				},
				ModelInfo: map[string]any{
					"general.architecture":       "llama",
					"llama.context_length":       float64(8192),
					"llama.embedding_length":     float64(4096),
					"llama.block_count":          float64(32),
					"llama.attention.head_count": float64(32),
				},
				Parameters: "temperature 0.7\nstop \"<|end|>\"",
				Capabilities: []model.Capability{
					model.CapabilityCompletion,
					model.CapabilityTools,
				},
				Tensors: []api.Tensor{
					{Name: "token_embd.weight", Type: "Q4_K", Shape: []uint64{4096, 32000}},
					{Name: "blk.0.attn_q.weight", Type: "Q4_K", Shape: []uint64{4096, 4096}},
					{Name: "output.weight", Type: "Q6_K", Shape: []uint64{4096, 32000}},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer server.Close()

	origHost := os.Getenv("OLLAMA_HOST")
	os.Setenv("OLLAMA_HOST", server.URL)
	defer os.Setenv("OLLAMA_HOST", origHost)

	devCmd := NewDevCommand()
	infoCmd, _, err := devCmd.Find([]string{"info"})
	if err != nil {
		t.Fatalf("failed to find info subcommand: %v", err)
	}

	infoCmd.SetContext(context.Background())

	// Test text output
	var out bytes.Buffer
	infoCmd.SetOut(&out)

	if err := DevInfoHandler(infoCmd, []string{"test-model"}); err != nil {
		t.Fatalf("DevInfoHandler error: %v", err)
	}

	// Test JSON flag
	_ = infoCmd.Flags().Set("json", "true")
	if err := DevInfoHandler(infoCmd, []string{"aabbcc112233"}); err != nil {
		t.Fatalf("DevInfoHandler with --json error: %v", err)
	}
}

func TestDevInfoHandler_NotFound(t *testing.T) {
	cmd := NewDevCommand()
	infoCmd, _, _ := cmd.Find([]string{"info"})
	infoCmd.SetContext(context.Background())

	err := DevInfoHandler(infoCmd, []string{"definitely-non-existent-model-xyz"})
	if err == nil {
		t.Fatalf("expected error for non existent model, got nil")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("expected error message to contain 'not found', got: %v", err)
	}
}

func TestVisualizerServer(t *testing.T) {
	mockOllama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/ps":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(api.ProcessResponse{})
		case "/api/generate":
			w.Header().Set("Content-Type", "application/x-ndjson")
			flusher, _ := w.(http.Flusher)
			chunk1 := map[string]any{"response": "Hello", "done": false}
			_ = json.NewEncoder(w).Encode(chunk1)
			if flusher != nil {
				flusher.Flush()
			}
			chunk2 := map[string]any{"response": " world", "done": true}
			_ = json.NewEncoder(w).Encode(chunk2)
			if flusher != nil {
				flusher.Flush()
			}
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer mockOllama.Close()

	targetURL, _ := http.NewRequest("GET", mockOllama.URL, nil)

	vs, err := NewVisualizerServer(0, targetURL.URL)
	if err != nil {
		t.Fatalf("NewVisualizerServer failed: %v", err)
	}

	// 1. Test HTML endpoint
	reqRoot := httptest.NewRequest("GET", "/", nil)
	recRoot := httptest.NewRecorder()
	vs.httpServer.Handler.ServeHTTP(recRoot, reqRoot)
	if recRoot.Code != http.StatusOK || !strings.Contains(recRoot.Body.String(), "NEURAL RADAR") {
		t.Fatalf("expected visualizer HTML, got status: %d", recRoot.Code)
	}

	// 2. Test status endpoint
	reqStatus := httptest.NewRequest("GET", "/api/dev/status", nil)
	recStatus := httptest.NewRecorder()
	vs.httpServer.Handler.ServeHTTP(recStatus, reqStatus)
	if recStatus.Code != http.StatusOK {
		t.Fatalf("expected status 200, got: %d", recStatus.Code)
	}

	// 3. Test inference proxy and event interception
	reqGen := httptest.NewRequest("POST", "/api/generate", strings.NewReader(`{"model":"test","prompt":"hi"}`))
	recGen := httptest.NewRecorder()
	vs.httpServer.Handler.ServeHTTP(recGen, reqGen)
	if recGen.Code != http.StatusOK || !strings.Contains(recGen.Body.String(), "Hello") {
		t.Fatalf("expected proxied response 'Hello', got: %s", recGen.Body.String())
	}
}
