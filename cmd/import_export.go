package cmd

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/types/model"
)

// ExportHandler exports a local model's GGUF file to a specified path.
func ExportHandler(cmd *cobra.Command, args []string) error {
	if cmd.Context() == nil {
		cmd.SetContext(context.Background())
	}
	modelName := args[0]
	outputPath, _ := cmd.Flags().GetString("output")
	if outputPath == "" && len(args) > 1 {
		outputPath = args[1]
	}
	if outputPath == "" {
		safeName := strings.ReplaceAll(strings.ReplaceAll(modelName, "/", "_"), ":", "_")
		if !strings.HasSuffix(strings.ToLower(safeName), ".gguf") {
			safeName += ".gguf"
		}
		outputPath = safeName
	}

	n := model.ParseName(modelName)
	if !n.IsValid() {
		return fmt.Errorf("invalid model name: %s", modelName)
	}

	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		return fmt.Errorf("failed to find model manifest for %s: %w", modelName, err)
	}

	if mf.Format == "safetensors" {
		return fmt.Errorf("model %s is a safetensors model; export currently only supports GGUF models", modelName)
	}

	var modelLayers []manifest.Layer
	for _, l := range mf.Layers {
		if l.MediaType == "application/vnd.ollama.image.model" {
			modelLayers = append(modelLayers, l)
		}
	}

	if len(modelLayers) == 0 {
		return fmt.Errorf("model %s does not contain any GGUF model layers", modelName)
	}

	var totalSize int64
	var sourcePaths []string
	for _, l := range modelLayers {
		bp, err := manifest.BlobsPath(l.Digest)
		if err != nil {
			return fmt.Errorf("failed to locate blob for layer %s: %w", l.Digest, err)
		}
		fi, err := os.Stat(bp)
		if err != nil {
			return fmt.Errorf("model weight file %s missing: %w", bp, err)
		}
		totalSize += fi.Size()
		sourcePaths = append(sourcePaths, bp)
	}

	// Verify magic bytes of the first layer
	f0, err := os.Open(sourcePaths[0])
	if err != nil {
		return err
	}
	magic := make([]byte, 4)
	if _, err := io.ReadFull(f0, magic); err != nil {
		f0.Close()
		return fmt.Errorf("failed to read model header: %w", err)
	}
	f0.Close()
	if string(magic) != "GGUF" {
		return fmt.Errorf("model layer %s is not a valid GGUF file (header is %q)", sourcePaths[0], string(magic))
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	outDir := filepath.Dir(outputPath)
	if outDir != "" && outDir != "." {
		if err := os.MkdirAll(outDir, 0o755); err != nil {
			return fmt.Errorf("failed to create output directory: %w", err)
		}
	}

	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create destination file %s: %w", outputPath, err)
	}
	defer outFile.Close()

	bar := progress.NewBar(fmt.Sprintf("exporting %s to %s", modelName, filepath.Base(outputPath)), totalSize, 0)
	p.Add("export", bar)

	buf := make([]byte, 2*1024*1024)
	var written int64
	for _, src := range sourcePaths {
		sf, err := os.Open(src)
		if err != nil {
			return fmt.Errorf("failed to open source %s: %w", src, err)
		}
		for {
			nr, rerr := sf.Read(buf)
			if nr > 0 {
				nw, werr := outFile.Write(buf[:nr])
				if nw > 0 {
					written += int64(nw)
					bar.Set(written)
				}
				if werr != nil {
					sf.Close()
					return fmt.Errorf("write error: %w", werr)
				}
			}
			if rerr != nil {
				if errors.Is(rerr, io.EOF) {
					break
				}
				sf.Close()
				return fmt.Errorf("read error: %w", rerr)
			}
		}
		sf.Close()
	}

	p.Stop()
	fmt.Printf("Successfully exported %s to %s (%s)\n", modelName, outputPath, format.HumanBytes(written))
	return nil
}

// ImportHandler imports a GGUF file into Ollama under a specified model name.
func ImportHandler(cmd *cobra.Command, args []string) error {
	if cmd.Context() == nil {
		cmd.SetContext(context.Background())
	}
	var modelName, ggufPath string

	if len(args) == 1 {
		ggufPath = args[0]
		base := filepath.Base(ggufPath)
		ext := filepath.Ext(base)
		modelName = strings.TrimSuffix(base, ext)
		modelName = strings.ToLower(modelName)
	} else if len(args) >= 2 {
		arg0, arg1 := args[0], args[1]
		_, err0 := os.Stat(arg0)
		_, err1 := os.Stat(arg1)

		if err0 == nil && (strings.HasSuffix(strings.ToLower(arg0), ".gguf") || err1 != nil) {
			ggufPath = arg0
			modelName = arg1
		} else {
			modelName = arg0
			ggufPath = arg1
		}
	} else {
		return errors.New("import requires a model name and a GGUF file path. Usage: ollama import <model> <file.gguf>")
	}

	name := model.ParseName(modelName)
	if !name.IsValid() {
		return fmt.Errorf("invalid model name: %s", modelName)
	}

	fi, err := os.Stat(ggufPath)
	if err != nil {
		return fmt.Errorf("failed to open GGUF file: %w", err)
	}
	if fi.IsDir() {
		return fmt.Errorf("%s is a directory, expected a GGUF file", ggufPath)
	}

	f, err := os.Open(ggufPath)
	if err != nil {
		return err
	}
	magic := make([]byte, 4)
	if _, err := io.ReadFull(f, magic); err != nil {
		f.Close()
		return fmt.Errorf("failed to read file header: %w", err)
	}
	f.Close()
	if string(magic) != "GGUF" {
		return fmt.Errorf("%s is not a valid GGUF file (header is %q, expected 'GGUF')", ggufPath, string(magic))
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	h := sha256.New()
	f, err = os.Open(ggufPath)
	if err != nil {
		return err
	}
	defer f.Close()

	hashBar := progress.NewBar(fmt.Sprintf("calculating digest for %s", filepath.Base(ggufPath)), fi.Size(), 0)
	p.Add("hash", hashBar)

	buf := make([]byte, 2*1024*1024)
	var hashedBytes int64
	for {
		nr, rerr := f.Read(buf)
		if nr > 0 {
			h.Write(buf[:nr])
			hashedBytes += int64(nr)
			hashBar.Set(hashedBytes)
		}
		if rerr != nil {
			if errors.Is(rerr, io.EOF) {
				break
			}
			return rerr
		}
	}
	digest := fmt.Sprintf("sha256:%x", h.Sum(nil))
	p.Stop()

	if err := checkServerHeartbeat(cmd, args); err != nil {
		return err
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	local := sharedBlobStore(cmd.Context(), client)

	p = progress.NewProgress(os.Stderr)
	defer p.Stop()

	if _, err := createBlob(cmd, client, ggufPath, digest, p, local); err != nil {
		return err
	}

	status := "registering model"
	spinner := progress.NewSpinner(status)
	p.Add(status, spinner)

	req := &api.CreateRequest{
		Model: modelName,
		Files: map[string]string{
			filepath.Base(ggufPath): digest,
		},
	}

	bars := make(map[string]*progress.Bar)
	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			bar, ok := bars[resp.Digest]
			if !ok {
				msg := resp.Status
				if msg == "" {
					msg = fmt.Sprintf("processing %s...", resp.Digest[7:19])
				}
				bar = progress.NewBar(msg, resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}
			bar.Set(resp.Completed)
		} else if status != resp.Status {
			spinner.Stop()
			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}
		return nil
	}

	if err := client.Create(cmd.Context(), req, fn); err != nil {
		return err
	}

	p.Stop()
	fmt.Printf("Successfully imported %s from %s\n", modelName, ggufPath)
	return nil
}
