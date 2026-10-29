package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestExportHandler_MissingModel(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	cmd := &cobra.Command{}
	cmd.Flags().StringP("output", "o", "", "")

	err := ExportHandler(cmd, []string{"nonexistent-model"})
	if err == nil || !strings.Contains(err.Error(), "failed to find model manifest") {
		t.Fatalf("expected manifest not found error, got: %v", err)
	}
}

func TestExportHandler_Success(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	ggufContent := "GGUFfake-weight-data-for-testing"
	layer, err := manifest.NewLayer(strings.NewReader(ggufContent), "application/vnd.ollama.image.model")
	if err != nil {
		t.Fatalf("failed to create layer: %v", err)
	}

	configLayer, err := manifest.NewLayer(strings.NewReader(`{"model_format":"gguf"}`), "application/vnd.docker.container.image.v1+json")
	if err != nil {
		t.Fatalf("failed to create config: %v", err)
	}

	modelName := model.ParseName("test-export-model:latest")
	if err := manifest.WriteManifest(modelName, configLayer, []manifest.Layer{layer}); err != nil {
		t.Fatalf("failed to write manifest: %v", err)
	}

	outPath := filepath.Join(t.TempDir(), "exported.gguf")
	cmd := &cobra.Command{}
	cmd.Flags().StringP("output", "o", "", "")

	if err := ExportHandler(cmd, []string{"test-export-model", outPath}); err != nil {
		t.Fatalf("ExportHandler failed: %v", err)
	}

	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("failed to read exported file: %v", err)
	}

	if string(data) != ggufContent {
		t.Fatalf("exported data mismatch: got %q, want %q", string(data), ggufContent)
	}
}

func TestImportHandler_Validation(t *testing.T) {
	cmd := &cobra.Command{}

	// Missing file
	if err := ImportHandler(cmd, []string{"mymodel", "non-existent-file.gguf"}); err == nil {
		t.Fatal("expected error for non-existent file, got nil")
	}

	// Invalid GGUF header
	tmpDir := t.TempDir()
	invalidFile := filepath.Join(tmpDir, "invalid.gguf")
	if err := os.WriteFile(invalidFile, []byte("NOTGGUF_DATA"), 0o644); err != nil {
		t.Fatal(err)
	}

	err := ImportHandler(cmd, []string{"mymodel", invalidFile})
	if err == nil || !strings.Contains(err.Error(), "not a valid GGUF file") {
		t.Fatalf("expected invalid GGUF error, got: %v", err)
	}
}

func TestImportHandler_Success(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	// Create a valid dummy GGUF file
	tmpDir := t.TempDir()
	ggufFile := filepath.Join(tmpDir, "my-model.gguf")
	ggufContent := []byte("GGUFtest-content-for-import")
	if err := os.WriteFile(ggufFile, ggufContent, 0o644); err != nil {
		t.Fatal(err)
	}

	var createdModel string
	var createdFiles map[string]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodHead && strings.HasPrefix(r.URL.Path, "/api/blobs/"):
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodGet && r.URL.Path == "/api/version":
			_ = json.NewEncoder(w).Encode(map[string]string{"version": "0.0.0"})
		case r.Method == http.MethodPost && r.URL.Path == "/api/create":
			var req api.CreateRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			createdModel = req.Model
			createdFiles = req.Files
			w.Header().Set("Content-Type", "application/x-ndjson")
			_ = json.NewEncoder(w).Encode(api.ProgressResponse{Status: "success"})
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	cmd := &cobra.Command{}
	if err := ImportHandler(cmd, []string{"imported-model", ggufFile}); err != nil {
		t.Fatalf("ImportHandler failed: %v", err)
	}

	if createdModel != "imported-model" {
		t.Fatalf("expected created model 'imported-model', got %q", createdModel)
	}
	if len(createdFiles) != 1 {
		t.Fatalf("expected 1 file in create request, got %v", createdFiles)
	}
}
