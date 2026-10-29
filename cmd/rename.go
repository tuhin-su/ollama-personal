package cmd

import (
	"context"
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// RenameHandler renames a model identified by name or ID to a new name.
func RenameHandler(cmd *cobra.Command, args []string) error {
	ctx := cmd.Context()
	if ctx == nil {
		ctx = context.Background()
	}

	source := args[0]
	dest := args[1]

	destName := model.ParseName(dest)
	if !destName.IsValid() {
		return fmt.Errorf("invalid destination model name: %s", dest)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	// Resolve source if an ID or partial name was provided
	resolvedSource, err := resolveModelNameOrID(ctx, client, source)
	if err != nil {
		return err
	}

	// Validate source and destination are not identical
	if strings.EqualFold(resolvedSource, dest) || strings.EqualFold(resolvedSource, destName.DisplayShortest()) {
		return fmt.Errorf("source and destination are the same: %s", dest)
	}

	// 1. Copy the model to destination name
	copyReq := &api.CopyRequest{
		Source:      resolvedSource,
		Destination: dest,
	}
	if _, err := client.Copy(ctx, copyReq); err != nil {
		return fmt.Errorf("failed to copy model from '%s' to '%s': %w", resolvedSource, dest, err)
	}

	// 2. Unload the source model if currently active in memory
	_ = loadOrUnloadModel(cmd, &runOptions{
		Model:     resolvedSource,
		KeepAlive: &api.Duration{Duration: 0},
	})

	// 3. Delete the original model
	deleteReq := &api.DeleteRequest{
		Name: resolvedSource,
	}
	if err := client.Delete(ctx, deleteReq); err != nil {
		return fmt.Errorf("copied to '%s', but failed to remove original '%s': %w", dest, resolvedSource, err)
	}

	fmt.Printf("renamed '%s' to '%s'\n", resolvedSource, dest)
	return nil
}

// resolveModelNameOrID matches an input against existing model names or their ID/digest prefix.
func resolveModelNameOrID(ctx context.Context, client *api.Client, input string) (string, error) {
	models, err := client.List(ctx)
	if err != nil {
		return input, nil
	}

	cleanInput := strings.TrimPrefix(strings.ToLower(input), "sha256:")
	var idMatches []string

	for _, m := range models.Models {
		// Exact name match
		if strings.EqualFold(m.Name, input) || strings.EqualFold(m.Model, input) {
			return m.Name, nil
		}

		cleanDigest := strings.TrimPrefix(strings.ToLower(m.Digest), "sha256:")
		if strings.HasPrefix(cleanDigest, cleanInput) {
			idMatches = append(idMatches, m.Name)
		}
	}

	if len(idMatches) > 0 {
		return idMatches[0], nil
	}

	return input, nil
}
