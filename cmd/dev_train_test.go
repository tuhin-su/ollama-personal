package cmd

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCalculateLoRASize(t *testing.T) {
	cfg := LoRAConfig{
		Rank:          8,
		Alpha:         16,
		TargetModules: []string{"q_proj", "v_proj"},
	}

	blockCount := 32
	embedLen := 4096

	params, sizeBytes := calculateLoRASize(blockCount, embedLen, cfg)

	// 32 layers * 2 target modules * (2 * 4096 * 8) = 4,194,304 parameters
	expectedParams := uint64(32 * 2 * (2 * 4096 * 8))
	if params != expectedParams {
		t.Errorf("expected %d params, got %d", expectedParams, params)
	}

	// 4,194,304 * 2 bytes + 32KB = 8,421,376 bytes (~8.03 MB)
	if sizeBytes < 8*1024*1024 || sizeBytes > 9*1024*1024 {
		t.Errorf("expected size ~8.4MB, got %d bytes (%s)", sizeBytes, formatBytes(int64(sizeBytes)))
	}
}

func TestWriteGGUFLoRAAdapter(t *testing.T) {
	tempDir := t.TempDir()
	adapterPath := filepath.Join(tempDir, "test_adapter.gguf")

	err := writeGGUFLoRAAdapter(adapterPath, "llama", 8, 16, 4, 1024)
	if err != nil {
		t.Fatalf("writeGGUFLoRAAdapter failed: %v", err)
	}

	stat, err := os.Stat(adapterPath)
	if err != nil {
		t.Fatalf("failed to stat generated adapter: %v", err)
	}

	if stat.Size() == 0 {
		t.Fatalf("adapter file is empty")
	}

	// Verify header starts with GGUF
	data, err := os.ReadFile(adapterPath)
	if err != nil {
		t.Fatalf("failed to read adapter: %v", err)
	}

	if string(data[:4]) != "GGUF" {
		t.Fatalf("expected GGUF magic header, got %q", string(data[:4]))
	}
}

func TestDetectGPUDevice(t *testing.T) {
	gpu := DetectGPUDevice()
	if gpu.Type == "" {
		t.Fatalf("expected non-empty GPU type")
	}
	t.Logf("Detected Device: %s (%s, VRAM: %d MB)", gpu.Name, gpu.Type, gpu.VRAMTotalMB)
}
