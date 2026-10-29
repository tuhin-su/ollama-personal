package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
)

func TestRenameHandler_InvalidDestination(t *testing.T) {
	cmd := &cobra.Command{}
	err := RenameHandler(cmd, []string{"old-model", "invalid/name/extra/slashes/here"})
	if err == nil || !strings.Contains(err.Error(), "invalid destination model name") {
		t.Fatalf("expected invalid destination error, got: %v", err)
	}
}

func TestRenameHandler_SameSourceAndDest(t *testing.T) {
	cmd := &cobra.Command{}
	err := RenameHandler(cmd, []string{"same-model", "same-model"})
	if err == nil || !strings.Contains(err.Error(), "source and destination are the same") {
		t.Fatalf("expected same source and destination error, got: %v", err)
	}
}

func TestRenameHandler_ByName(t *testing.T) {
	var copiedSource, copiedDest string
	var deletedModel string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/api/version":
			_ = json.NewEncoder(w).Encode(map[string]string{"version": "0.0.0"})
		case r.Method == http.MethodGet && r.URL.Path == "/api/tags":
			resp := api.ListResponse{
				Models: []api.ListModelResponse{
					{Name: "old-model:latest", Model: "old-model", Digest: "sha256:1234567890ab"},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		case r.Method == http.MethodPost && r.URL.Path == "/api/copy":
			var req api.CopyRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			copiedSource = req.Source
			copiedDest = req.Destination
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodPost && r.URL.Path == "/api/generate":
			// For unload model
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodDelete && r.URL.Path == "/api/delete":
			var req api.DeleteRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			deletedModel = req.Name
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	cmd := &cobra.Command{}
	if err := RenameHandler(cmd, []string{"old-model", "new-model"}); err != nil {
		t.Fatalf("RenameHandler failed: %v", err)
	}

	if copiedSource != "old-model:latest" {
		t.Fatalf("expected copied source 'old-model:latest', got %q", copiedSource)
	}
	if copiedDest != "new-model" {
		t.Fatalf("expected copied dest 'new-model', got %q", copiedDest)
	}
	if deletedModel != "old-model:latest" {
		t.Fatalf("expected deleted model 'old-model:latest', got %q", deletedModel)
	}
}

func TestRenameHandler_ByID(t *testing.T) {
	var copiedSource, copiedDest string
	var deletedModel string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/api/version":
			_ = json.NewEncoder(w).Encode(map[string]string{"version": "0.0.0"})
		case r.Method == http.MethodGet && r.URL.Path == "/api/tags":
			resp := api.ListResponse{
				Models: []api.ListModelResponse{
					{Name: "custom-model:latest", Model: "custom-model", Digest: "sha256:d6896264e107fedcba987654"},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		case r.Method == http.MethodPost && r.URL.Path == "/api/copy":
			var req api.CopyRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			copiedSource = req.Source
			copiedDest = req.Destination
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodPost && r.URL.Path == "/api/generate":
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodDelete && r.URL.Path == "/api/delete":
			var req api.DeleteRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			deletedModel = req.Name
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", server.URL)

	cmd := &cobra.Command{}
	// Pass short 12-char ID matching the model digest
	if err := RenameHandler(cmd, []string{"d6896264e107", "renamed-model"}); err != nil {
		t.Fatalf("RenameHandler by ID failed: %v", err)
	}

	if copiedSource != "custom-model:latest" {
		t.Fatalf("expected resolved copied source 'custom-model:latest', got %q", copiedSource)
	}
	if copiedDest != "renamed-model" {
		t.Fatalf("expected copied dest 'renamed-model', got %q", copiedDest)
	}
	if deletedModel != "custom-model:latest" {
		t.Fatalf("expected deleted model 'custom-model:latest', got %q", deletedModel)
	}
}
