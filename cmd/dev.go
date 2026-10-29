package cmd

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/mattn/go-runewidth"
	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// NewDevCommand creates the 'dev' command group for internal inspection and developer tools.
func NewDevCommand() *cobra.Command {
	devCmd := &cobra.Command{
		Use:   "dev",
		Short: "Developer inspection and debugging commands",
	}

	infoCmd := &cobra.Command{
		Use:   "info MODEL",
		Short: "Show deep inspection information for a model (parameters, layers, tensors, metadata)",
		Args:  cobra.ExactArgs(1),
		RunE:  DevInfoHandler,
	}

	infoCmd.Flags().BoolP("all", "a", false, "Show all deep inspection sections")
	infoCmd.Flags().BoolP("layers", "l", false, "Show layer-by-layer architectural breakdown")
	infoCmd.Flags().BoolP("tensors", "t", false, "Show detailed tensor table")
	infoCmd.Flags().BoolP("parameters", "p", false, "Show model parameters and hyperparameters")
	infoCmd.Flags().BoolP("metadata", "m", false, "Show full GGUF key-value metadata")
	infoCmd.Flags().BoolP("storage", "s", false, "Show on-disk blob storage and manifest details")
	infoCmd.Flags().BoolP("json", "j", false, "Output deep model information in JSON format")

	uiCmd := &cobra.Command{
		Use:     "ui",
		Aliases: []string{"visualizer", "viz"},
		Short:   "Launch local web visualizer to monitor neural network layers, active neurons, and connections in real time",
		Args:    cobra.NoArgs,
		RunE:    DevUIHandler,
	}

	uiCmd.Flags().IntP("port", "p", 11435, "Port for the local visualizer web UI and listener")
	uiCmd.Flags().String("target", "http://127.0.0.1:11434", "Target Ollama server URL")
	uiCmd.Flags().Bool("no-browser", false, "Do not automatically open browser on startup")

	memoryCmd := NewDevMemoryCommand()
	trainCmd := NewDevTrainCommand()

	devCmd.AddCommand(infoCmd, uiCmd, memoryCmd, trainCmd)
	return devCmd
}

// DevModelInfo holds complete deep inspection details for a model.
type DevModelInfo struct {
	Name         string            `json:"name"`
	ID           string            `json:"id,omitempty"`
	Format       string            `json:"format"`
	Architecture string            `json:"architecture"`
	Family       string            `json:"family,omitempty"`
	ParameterSize string           `json:"parameter_size"`
	ExactParams  uint64            `json:"parameter_count"`
	Quantization string            `json:"quantization"`
	Runner       string            `json:"runner,omitempty"`
	ModifiedAt   string            `json:"modified_at,omitempty"`
	Dimensions   ModelDimensions   `json:"dimensions"`
	Capabilities []string          `json:"capabilities,omitempty"`
	Runtime      map[string]string `json:"runtime_parameters,omitempty"`
	Storage      *StorageInfo      `json:"storage,omitempty"`
	QuantDist    []QuantSummary    `json:"quantization_distribution,omitempty"`
	Layers       []LayerSummary    `json:"layers,omitempty"`
	Tensors      []DevTensor       `json:"tensors,omitempty"`
	Metadata     map[string]any    `json:"metadata,omitempty"`
}

// ModelDimensions stores dimensional and architectural properties of the model.
type ModelDimensions struct {
	ContextLength     uint64  `json:"context_length,omitempty"`
	EmbeddingLength   uint64  `json:"embedding_length,omitempty"`
	BlockCount        uint64  `json:"block_count,omitempty"`
	AttentionHeads    uint64  `json:"attention_heads,omitempty"`
	AttentionHeadsKV  uint64  `json:"attention_heads_kv,omitempty"`
	GQARatio          string  `json:"gqa_ratio,omitempty"`
	FeedForwardLength uint64  `json:"feed_forward_length,omitempty"`
	VocabSize         uint64  `json:"vocab_size,omitempty"`
	RoPEFreqBase      float64 `json:"rope_freq_base,omitempty"`
	RoPEDimCount      uint64  `json:"rope_dimension_count,omitempty"`
	LayerNormRMSEps   float64 `json:"layer_norm_rms_epsilon,omitempty"`
}

// StorageInfo represents local manifest and blob storage on disk.
type StorageInfo struct {
	ManifestPath string         `json:"manifest_path,omitempty"`
	Digest       string         `json:"digest,omitempty"`
	TotalSize    uint64         `json:"total_size"`
	Layers       []StorageLayer `json:"layers"`
}

// StorageLayer describes an individual layer stored on disk.
type StorageLayer struct {
	MediaType string `json:"media_type"`
	Digest    string `json:"digest"`
	Size      uint64 `json:"size"`
	Path      string `json:"path,omitempty"`
	Exists    bool   `json:"exists"`
}

// QuantSummary summarizes parameters for each quantization type.
type QuantSummary struct {
	Type         string  `json:"type"`
	TensorCount  int     `json:"tensor_count"`
	TotalParams  uint64  `json:"total_params"`
	PercentParam float64 `json:"percent_params"`
}

// LayerSummary summarizes a structural group/block of layers.
type LayerSummary struct {
	Name        string   `json:"name"`
	Components  []string `json:"components"`
	TensorCount int      `json:"tensor_count"`
	ParamCount  uint64   `json:"param_count"`
	DataTypes   []string `json:"data_types"`
}

// DevTensor holds metadata for a single tensor.
type DevTensor struct {
	Index      int      `json:"index"`
	Name       string   `json:"name"`
	Type       string   `json:"type"`
	Shape      []uint64 `json:"shape"`
	ParamCount uint64   `json:"param_count"`
}

// DevInfoHandler handles 'ollama dev info MODEL'.
func DevInfoHandler(cmd *cobra.Command, args []string) error {
	ctx := cmd.Context()
	if ctx == nil {
		ctx = context.Background()
	}

	asJSON, _ := cmd.Flags().GetBool("json")
	showAll, _ := cmd.Flags().GetBool("all")
	showLayers, _ := cmd.Flags().GetBool("layers")
	showTensors, _ := cmd.Flags().GetBool("tensors")
	showParams, _ := cmd.Flags().GetBool("parameters")
	showMetadata, _ := cmd.Flags().GetBool("metadata")
	showStorage, _ := cmd.Flags().GetBool("storage")

	// If no specific section flags were set, show everything by default
	if !showLayers && !showTensors && !showParams && !showMetadata && !showStorage {
		showAll = true
	}

	modelInput := args[0]
	devInfo, err := loadDevModelInfo(ctx, modelInput)
	if err != nil {
		return err
	}

	if asJSON {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		return encoder.Encode(devInfo)
	}

	return renderDevInfoText(os.Stdout, devInfo, showAll, showLayers, showTensors, showParams, showMetadata, showStorage)
}

// loadDevModelInfo fetches deep model information from the client or offline files.
func loadDevModelInfo(ctx context.Context, input string) (*DevModelInfo, error) {
	client, clientErr := api.ClientFromEnvironment()

	// 1. Resolve model name or ID
	resolvedName := input
	if clientErr == nil {
		if r, err := resolveModelNameOrID(ctx, client, input); err == nil && r != "" {
			resolvedName = r
		}
	}
	if resolvedName == input {
		if r, err := resolveOfflineModelNameOrID(input); err == nil && r != "" {
			resolvedName = r
		}
	}

	parsedName := model.ParseName(resolvedName)

	var resp *api.ShowResponse
	var fetchErr error

	if clientErr == nil {
		req := &api.ShowRequest{Name: resolvedName, Verbose: true}
		resp, fetchErr = client.Show(ctx, req)
	}

	// If client fetch failed, attempt direct offline inspection via manifest & GGUF
	if fetchErr != nil || resp == nil {
		offlineResp, offlineErr := loadOfflineDevInfo(parsedName)
		if offlineErr != nil {
			if errors.Is(offlineErr, os.ErrNotExist) || strings.Contains(strings.ToLower(offlineErr.Error()), "not exist") {
				return nil, fmt.Errorf("model '%s' not found", input)
			}
			if fetchErr != nil && !strings.Contains(fetchErr.Error(), "connection refused") {
				return nil, fmt.Errorf("unable to inspect model '%s': %v (offline inspection failed: %w)", input, fetchErr, offlineErr)
			}
			return nil, fmt.Errorf("unable to inspect model '%s': %w", input, offlineErr)
		}
		resp = offlineResp
	}

	// 2. Build DevModelInfo struct
	devInfo := &DevModelInfo{
		Name:         resolvedName,
		Format:       resp.Details.Format,
		Architecture: resp.Details.Family,
		Family:       resp.Details.Family,
		Quantization: resp.Details.QuantizationLevel,
		Runner:       resp.Details.Runner,
		Metadata:     resp.ModelInfo,
	}

	if !resp.ModifiedAt.IsZero() {
		devInfo.ModifiedAt = resp.ModifiedAt.Format("2006-01-02 15:04:05 -0700 MST")
	}

	for _, cap := range resp.Capabilities {
		devInfo.Capabilities = append(devInfo.Capabilities, cap.String())
	}

	// Extract runtime parameters
	devInfo.Runtime = make(map[string]string)
	if resp.Parameters != "" {
		scanner := bufio.NewScanner(strings.NewReader(resp.Parameters))
		for scanner.Scan() {
			fields := strings.Fields(scanner.Text())
			if len(fields) >= 2 {
				devInfo.Runtime[fields[0]] = strings.Join(fields[1:], " ")
			}
		}
	}

	// Populate tensors and calculate exact parameter count
	var totalParams uint64
	devInfo.Tensors = make([]DevTensor, 0, len(resp.Tensors))
	for i, t := range resp.Tensors {
		var numElements uint64 = 1
		for _, dim := range t.Shape {
			if dim > 0 {
				numElements *= dim
			}
		}
		totalParams += numElements
		devInfo.Tensors = append(devInfo.Tensors, DevTensor{
			Index:      i + 1,
			Name:       t.Name,
			Type:       t.Type,
			Shape:      t.Shape,
			ParamCount: numElements,
		})
	}

	devInfo.ExactParams = totalParams
	if devInfo.ExactParams > 0 {
		devInfo.ParameterSize = format.HumanNumber(devInfo.ExactParams)
	} else if resp.Details.ParameterSize != "" {
		devInfo.ParameterSize = resp.Details.ParameterSize
	}

	// 3. Compute Dimensions from ModelInfo
	devInfo.Dimensions = computeDimensions(resp.ModelInfo, devInfo.Tensors)
	if devInfo.Dimensions.BlockCount > 0 && devInfo.Architecture == "" {
		if arch, ok := resp.ModelInfo["general.architecture"].(string); ok {
			devInfo.Architecture = arch
		}
	}

	// 4. Compute Quantization Distribution
	devInfo.QuantDist = computeQuantDistribution(devInfo.Tensors, totalParams)

	// 5. Compute Layer Architecture Summary
	devInfo.Layers = computeLayerSummary(devInfo.Tensors, devInfo.Dimensions.BlockCount)

	// 6. Compute Local Storage Info (manifest & blobs)
	devInfo.Storage = computeStorageInfo(parsedName)
	if devInfo.Storage != nil && devInfo.Storage.Digest != "" {
		cleanDigest := strings.TrimPrefix(devInfo.Storage.Digest, "sha256:")
		if len(cleanDigest) >= 12 {
			devInfo.ID = cleanDigest[:12]
		}
	}

	return devInfo, nil
}

// computeDimensions extracts architectural hyperparameters from ModelInfo and tensors.
func computeDimensions(modelInfo map[string]any, tensors []DevTensor) ModelDimensions {
	var dims ModelDimensions
	if modelInfo == nil {
		return dims
	}

	arch, _ := modelInfo["general.architecture"].(string)

	getUint := func(key string) uint64 {
		if v, ok := modelInfo[key]; ok {
			switch val := v.(type) {
			case uint64:
				return val
			case uint32:
				return uint64(val)
			case uint16:
				return uint64(val)
			case uint8:
				return uint64(val)
			case uint:
				return uint64(val)
			case int64:
				return uint64(val)
			case int32:
				return uint64(val)
			case int16:
				return uint64(val)
			case int8:
				return uint64(val)
			case int:
				return uint64(val)
			case float64:
				return uint64(val)
			case float32:
				return uint64(val)
			case string:
				if n, err := strconv.ParseUint(val, 10, 64); err == nil {
					return n
				}
			}
		}
		return 0
	}

	getFloat := func(key string) float64 {
		if v, ok := modelInfo[key]; ok {
			switch val := v.(type) {
			case float64:
				return val
			case float32:
				return float64(val)
			case int64:
				return float64(val)
			case int32:
				return float64(val)
			case int16:
				return float64(val)
			case int8:
				return float64(val)
			case int:
				return float64(val)
			case uint64:
				return float64(val)
			case uint32:
				return float64(val)
			case uint16:
				return float64(val)
			case uint8:
				return float64(val)
			case uint:
				return float64(val)
			case string:
				if f, err := strconv.ParseFloat(val, 64); err == nil {
					return f
				}
			}
		}
		return 0
	}

	if arch != "" {
		dims.ContextLength = getUint(arch + ".context_length")
		dims.EmbeddingLength = getUint(arch + ".embedding_length")
		dims.BlockCount = getUint(arch + ".block_count")
		dims.AttentionHeads = getUint(arch + ".attention.head_count")
		dims.AttentionHeadsKV = getUint(arch + ".attention.head_count_kv")
		dims.FeedForwardLength = getUint(arch + ".feed_forward_length")
		dims.RoPEFreqBase = getFloat(arch + ".rope.freq_base")
		dims.RoPEDimCount = getUint(arch + ".rope.dimension_count")
		dims.LayerNormRMSEps = getFloat(arch + ".attention.layer_norm_rms_epsilon")
	}

	// GQA Ratio
	if dims.AttentionHeads > 0 && dims.AttentionHeadsKV > 0 {
		if dims.AttentionHeads == dims.AttentionHeadsKV {
			dims.GQARatio = "1:1 (MHA)"
		} else {
			ratio := dims.AttentionHeads / dims.AttentionHeadsKV
			dims.GQARatio = fmt.Sprintf("%d:1 (GQA)", ratio)
		}
	}

	// Vocab size
	if v, ok := modelInfo["tokenizer.ggml.tokens"].([]any); ok && len(v) > 0 {
		dims.VocabSize = uint64(len(v))
	} else if arch != "" {
		dims.VocabSize = getUint(arch + ".vocab_size")
	}
	if dims.VocabSize == 0 {
		for _, t := range tensors {
			if strings.Contains(t.Name, "token_embd.weight") || strings.Contains(t.Name, "embed_tokens.weight") {
				for _, dim := range t.Shape {
					if dim > dims.EmbeddingLength && dim > dims.VocabSize {
						dims.VocabSize = dim
					}
				}
			}
		}
	}

	// Fallback for block count if not in metadata: derive from tensor names
	if dims.BlockCount == 0 {
		re := regexp.MustCompile(`(?:blk|layers)\.([0-9]+)\.`)
		var maxBlock int64 = -1
		for _, t := range tensors {
			matches := re.FindStringSubmatch(t.Name)
			if len(matches) >= 2 {
				if b, err := strconv.ParseInt(matches[1], 10, 64); err == nil && b > maxBlock {
					maxBlock = b
				}
			}
		}
		if maxBlock >= 0 {
			dims.BlockCount = uint64(maxBlock + 1)
		}
	}

	return dims
}

// computeQuantDistribution aggregates tensor counts and parameters by quantization type.
func computeQuantDistribution(tensors []DevTensor, totalParams uint64) []QuantSummary {
	typeStats := make(map[string]*QuantSummary)

	for _, t := range tensors {
		typeName := t.Type
		if typeName == "" {
			typeName = "UNKNOWN"
		}
		stat, exists := typeStats[typeName]
		if !exists {
			stat = &QuantSummary{Type: typeName}
			typeStats[typeName] = stat
		}
		stat.TensorCount++
		stat.TotalParams += t.ParamCount
	}

	var results []QuantSummary
	for _, stat := range typeStats {
		if totalParams > 0 {
			stat.PercentParam = (float64(stat.TotalParams) / float64(totalParams)) * 100.0
		}
		results = append(results, *stat)
	}

	// Sort by total params descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].TotalParams > results[j].TotalParams
	})

	return results
}

// computeLayerSummary groups tensors into architectural layer blocks.
func computeLayerSummary(tensors []DevTensor, blockCount uint64) []LayerSummary {
	blockRe := regexp.MustCompile(`^(?:blk|model\.layers|layers)\.([0-9]+)\.(.*)`)
	embedRe := regexp.MustCompile(`(?i)(token_embd|position_embd|model\.embed_tokens|embed_tokens)`)
	outputRe := regexp.MustCompile(`(?i)(output_norm|output\.weight|model\.norm|lm_head)`)
	projectorRe := regexp.MustCompile(`(?i)^(v\.blk|mm|v\.embed|projector)`)

	type blockData struct {
		name       string
		order      int
		components map[string]struct{}
		tensors    int
		params     uint64
		dataTypes  map[string]struct{}
	}

	groups := make(map[string]*blockData)

	getOrCreate := func(key string, name string, order int) *blockData {
		bd, exists := groups[key]
		if !exists {
			bd = &blockData{
				name:       name,
				order:      order,
				components: make(map[string]struct{}),
				dataTypes:  make(map[string]struct{}),
			}
			groups[key] = bd
		}
		return bd
	}

	for _, t := range tensors {
		var bd *blockData

		if m := blockRe.FindStringSubmatch(t.Name); len(m) >= 3 {
			bIdx, _ := strconv.Atoi(m[1])
			bd = getOrCreate(fmt.Sprintf("block_%04d", bIdx), fmt.Sprintf("Block %d", bIdx), 1000+bIdx)
			sub := m[2]
			switch {
			case strings.Contains(sub, "attn_q_norm"):
				bd.components["Attn Q-Norm"] = struct{}{}
			case strings.Contains(sub, "attn_k_norm"):
				bd.components["Attn K-Norm"] = struct{}{}
			case strings.Contains(sub, "attn_q"):
				bd.components["Attn Q"] = struct{}{}
			case strings.Contains(sub, "attn_k"):
				bd.components["Attn K"] = struct{}{}
			case strings.Contains(sub, "attn_v"):
				bd.components["Attn V"] = struct{}{}
			case strings.Contains(sub, "attn_output") || strings.Contains(sub, "attn_proj"):
				bd.components["Attn Out"] = struct{}{}
			case strings.Contains(sub, "attn_norm") || strings.Contains(sub, "input_layernorm"):
				bd.components["Attn Norm"] = struct{}{}
			case strings.Contains(sub, "ffn_gate_exps"):
				bd.components["MoE Gate Exp"] = struct{}{}
			case strings.Contains(sub, "ffn_up_exps"):
				bd.components["MoE Up Exp"] = struct{}{}
			case strings.Contains(sub, "ffn_down_exps"):
				bd.components["MoE Down Exp"] = struct{}{}
			case strings.Contains(sub, "ffn_gate"):
				bd.components["FFN Gate"] = struct{}{}
			case strings.Contains(sub, "ffn_up"):
				bd.components["FFN Up"] = struct{}{}
			case strings.Contains(sub, "ffn_down"):
				bd.components["FFN Down"] = struct{}{}
			case strings.Contains(sub, "ffn_norm") || strings.Contains(sub, "post_attention_layernorm"):
				bd.components["FFN Norm"] = struct{}{}
			default:
				bd.components[sub] = struct{}{}
			}
		} else if embedRe.MatchString(t.Name) {
			bd = getOrCreate("00_embed", "Embedding", 10)
			bd.components[t.Name] = struct{}{}
		} else if outputRe.MatchString(t.Name) {
			bd = getOrCreate("99_output", "Output / LM Head", 90000)
			bd.components[t.Name] = struct{}{}
		} else if projectorRe.MatchString(t.Name) {
			bd = getOrCreate("98_projector", "Projector / Vision", 80000)
			bd.components[t.Name] = struct{}{}
		} else {
			bd = getOrCreate("999_other", "Other / Global", 99999)
			bd.components[t.Name] = struct{}{}
		}

		bd.tensors++
		bd.params += t.ParamCount
		if t.Type != "" {
			bd.dataTypes[t.Type] = struct{}{}
		}
	}

	var sortedBlocks []*blockData
	for _, bd := range groups {
		sortedBlocks = append(sortedBlocks, bd)
	}
	sort.Slice(sortedBlocks, func(i, j int) bool {
		return sortedBlocks[i].order < sortedBlocks[j].order
	})

	var result []LayerSummary
	for _, bd := range sortedBlocks {
		comps := make([]string, 0, len(bd.components))
		for c := range bd.components {
			comps = append(comps, c)
		}
		sort.Strings(comps)

		dtypes := make([]string, 0, len(bd.dataTypes))
		for d := range bd.dataTypes {
			dtypes = append(dtypes, d)
		}
		sort.Strings(dtypes)

		result = append(result, LayerSummary{
			Name:        bd.name,
			Components:  comps,
			TensorCount: bd.tensors,
			ParamCount:  bd.params,
			DataTypes:   dtypes,
		})
	}

	return result
}

// computeStorageInfo inspects local manifests and blob files on disk.
func computeStorageInfo(name model.Name) *StorageInfo {
	mf, err := manifest.FindNamedManifestForRunner(name, "")
	if err != nil {
		mf, err = manifest.ParseNamedManifest(name)
		if err != nil {
			return nil
		}
	}

	manifestPath, _ := manifest.ResolvePathForName(name)
	storage := &StorageInfo{
		ManifestPath: manifestPath,
		Digest:       mf.BlobDigest(),
	}

	var totalSize uint64
	for _, l := range mf.Layers {
		totalSize += uint64(l.Size)
		blobPath, _ := manifest.BlobsPath(l.Digest)
		exists := false
		if blobPath != "" {
			if _, statErr := os.Stat(blobPath); statErr == nil {
				exists = true
			}
		}

		shortType := strings.TrimPrefix(l.MediaType, "application/vnd.ollama.image.")
		storage.Layers = append(storage.Layers, StorageLayer{
			MediaType: shortType,
			Digest:    l.Digest,
			Size:      uint64(l.Size),
			Path:      blobPath,
			Exists:    exists,
		})
	}
	storage.TotalSize = totalSize

	return storage
}

// resolveOfflineModelNameOrID resolves names or short digest IDs from disk manifests.
func resolveOfflineModelNameOrID(input string) (string, error) {
	manifests, err := manifest.Manifests(true)
	if err != nil {
		return input, nil
	}

	cleanInput := strings.TrimPrefix(strings.ToLower(input), "sha256:")

	for name, mf := range manifests {
		if strings.EqualFold(name.DisplayShortest(), input) || strings.EqualFold(name.String(), input) {
			return name.DisplayShortest(), nil
		}
		cleanDigest := strings.TrimPrefix(strings.ToLower(mf.BlobDigest()), "sha256:")
		if strings.HasPrefix(cleanDigest, cleanInput) {
			return name.DisplayShortest(), nil
		}
	}

	return input, nil
}

// loadOfflineDevInfo loads model metadata directly from GGUF and manifest files without a running server.
func loadOfflineDevInfo(name model.Name) (*api.ShowResponse, error) {
	mf, err := manifest.FindNamedManifestForRunner(name, "")
	if err != nil {
		mf, err = manifest.ParseNamedManifest(name)
		if err != nil {
			return nil, err
		}
	}

	var modelLayer *manifest.Layer
	for _, l := range mf.Layers {
		if l.MediaType == "application/vnd.ollama.image.model" {
			modelLayer = &l
			break
		}
	}
	if modelLayer == nil {
		return nil, errors.New("model layer not found in manifest")
	}

	blobPath, err := manifest.BlobsPath(modelLayer.Digest)
	if err != nil {
		return nil, err
	}

	m, err := gguf.ReadModel(blobPath, 0)
	if err != nil {
		return nil, fmt.Errorf("read GGUF model: %w", err)
	}

	resp := &api.ShowResponse{
		Details: api.ModelDetails{
			Format:            mf.Format,
			Family:            m.KV().Architecture(),
			QuantizationLevel: m.KV().FileType().String(),
			Runner:            mf.Runner,
		},
		ModelInfo:  m.KV().Values(),
		ModifiedAt: mf.FileInfo().ModTime(),
	}

	tensorItems := m.Tensors().Items()
	resp.Tensors = make([]api.Tensor, 0, len(tensorItems))
	for _, tensor := range tensorItems {
		tensorType := tensor.Type.String()
		if tensorType != "unknown" {
			tensorType = strings.ToUpper(tensorType)
		}
		resp.Tensors = append(resp.Tensors, api.Tensor{
			Name:  tensor.Name,
			Type:  tensorType,
			Shape: tensor.Shape,
		})
	}

	return resp, nil
}

// renderDevInfoText prints a structured, rich terminal report.
func renderDevInfoText(w io.Writer, info *DevModelInfo, showAll, showLayers, showTensors, showParams, showMetadata, showStorage bool) error {
	tableRender := func(header string, rows func() [][]string) {
		fmt.Fprintln(w, " ", header)
		table := tablewriter.NewWriter(w)
		table.SetAlignment(tablewriter.ALIGN_LEFT)
		table.SetBorder(false)
		table.SetNoWhiteSpace(true)
		table.SetTablePadding("    ")
		table.SetColWidth(80)
		table.AppendBulk(rows())
		table.Render()
		fmt.Fprintln(w)
	}

	// 1. Overview & Architecture Dimensions
	tableRender("Model Overview", func() [][]string {
		rows := [][]string{
			{"", "Name", info.Name},
		}
		if info.ID != "" {
			rows = append(rows, []string{"", "ID", info.ID})
		}
		if info.Format != "" {
			rows = append(rows, []string{"", "Format", strings.ToUpper(info.Format)})
		}
		if info.Architecture != "" {
			rows = append(rows, []string{"", "Architecture", info.Architecture})
		}
		if info.ParameterSize != "" {
			paramStr := info.ParameterSize
			if info.ExactParams > 0 {
				paramStr = fmt.Sprintf("%s (%s params)", info.ParameterSize, formatComma(info.ExactParams))
			}
			rows = append(rows, []string{"", "Parameters", paramStr})
		}
		if info.Quantization != "" {
			rows = append(rows, []string{"", "Quantization", info.Quantization})
		}
		if info.Runner != "" {
			rows = append(rows, []string{"", "Runner", info.Runner})
		}
		if len(info.Capabilities) > 0 {
			rows = append(rows, []string{"", "Capabilities", strings.Join(info.Capabilities, ", ")})
		}
		if info.ModifiedAt != "" {
			rows = append(rows, []string{"", "Modified", info.ModifiedAt})
		}
		return rows
	})

	// Architecture Dimensions
	if info.Dimensions.BlockCount > 0 || info.Dimensions.ContextLength > 0 || info.Dimensions.EmbeddingLength > 0 {
		tableRender("Architecture Dimensions", func() [][]string {
			var rows [][]string
			if info.Dimensions.ContextLength > 0 {
				rows = append(rows, []string{"", "Context Length", formatComma(info.Dimensions.ContextLength)})
			}
			if info.Dimensions.EmbeddingLength > 0 {
				rows = append(rows, []string{"", "Embedding Length", formatComma(info.Dimensions.EmbeddingLength)})
			}
			if info.Dimensions.BlockCount > 0 {
				rows = append(rows, []string{"", "Block / Layer Count", strconv.FormatUint(info.Dimensions.BlockCount, 10)})
			}
			if info.Dimensions.AttentionHeads > 0 {
				rows = append(rows, []string{"", "Attention Heads (Q)", strconv.FormatUint(info.Dimensions.AttentionHeads, 10)})
			}
			if info.Dimensions.AttentionHeadsKV > 0 {
				headKVStr := strconv.FormatUint(info.Dimensions.AttentionHeadsKV, 10)
				if info.Dimensions.GQARatio != "" {
					headKVStr = fmt.Sprintf("%s (%s)", headKVStr, info.Dimensions.GQARatio)
				}
				rows = append(rows, []string{"", "KV Attention Heads", headKVStr})
			}
			if info.Dimensions.FeedForwardLength > 0 {
				rows = append(rows, []string{"", "Feed Forward Length", formatComma(info.Dimensions.FeedForwardLength)})
			}
			if info.Dimensions.VocabSize > 0 {
				rows = append(rows, []string{"", "Vocabulary Size", formatComma(info.Dimensions.VocabSize)})
			}
			if info.Dimensions.RoPEFreqBase > 0 {
				rows = append(rows, []string{"", "RoPE Frequency Base", fmt.Sprintf("%g", info.Dimensions.RoPEFreqBase)})
			}
			if info.Dimensions.RoPEDimCount > 0 {
				rows = append(rows, []string{"", "RoPE Dimension Count", strconv.FormatUint(info.Dimensions.RoPEDimCount, 10)})
			}
			if info.Dimensions.LayerNormRMSEps > 0 {
				rows = append(rows, []string{"", "RMS Norm Epsilon", fmt.Sprintf("%g", info.Dimensions.LayerNormRMSEps)})
			}
			return rows
		})
	}

	// 2. Parameters & Hyperparameters
	if (showAll || showParams) && len(info.Runtime) > 0 {
		tableRender("Runtime & Modelfile Parameters", func() [][]string {
			var rows [][]string
			keys := make([]string, 0, len(info.Runtime))
			for k := range info.Runtime {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				rows = append(rows, []string{"", k, info.Runtime[k]})
			}
			return rows
		})
	}

	// 3. Storage & Manifest Layers
	if (showAll || showStorage) && info.Storage != nil {
		fmt.Fprintf(w, "  Storage & Manifest\n")
		if info.Storage.ManifestPath != "" {
			fmt.Fprintf(w, "    Manifest: %s\n", info.Storage.ManifestPath)
		}
		if info.Storage.TotalSize > 0 {
			fmt.Fprintf(w, "    Total Size: %s (%s bytes)\n\n", format.HumanBytes(int64(info.Storage.TotalSize)), formatComma(info.Storage.TotalSize))
		}

		table := tablewriter.NewWriter(w)
		table.SetHeader([]string{"", "TYPE", "DIGEST", "SIZE", "STATUS", "FILE PATH"})
		table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
		table.SetAlignment(tablewriter.ALIGN_LEFT)
		table.SetBorder(false)
		table.SetNoWhiteSpace(true)
		table.SetTablePadding("    ")

		for _, l := range info.Storage.Layers {
			status := "missing"
			if l.Exists {
				status = "ok"
			}
			dig := l.Digest
			if len(dig) > 19 {
				dig = dig[:19] + "..."
			}
			table.Append([]string{"", l.MediaType, dig, format.HumanBytes(int64(l.Size)), status, l.Path})
		}
		table.Render()
		fmt.Fprintln(w)
	}

	// 4. Quantization Distribution
	if (showAll || showLayers) && len(info.QuantDist) > 0 {
		tableRender("Quantization Distribution", func() [][]string {
			var rows [][]string
			rows = append(rows, []string{"TYPE", "TENSORS", "TOTAL PARAMS", "% OF PARAMS"})
			for _, q := range info.QuantDist {
				rows = append(rows, []string{
					q.Type,
					strconv.Itoa(q.TensorCount),
					format.HumanNumber(q.TotalParams),
					fmt.Sprintf("%.2f%%", q.PercentParam),
				})
			}
			return rows
		})
	}

	// 5. Layer Architecture Breakdown
	if (showAll || showLayers) && len(info.Layers) > 0 {
		header := fmt.Sprintf("Layer Architecture Breakdown (%d layers)", len(info.Layers))
		tableRender(header, func() [][]string {
			var rows [][]string
			rows = append(rows, []string{"LAYER", "COMPONENTS", "TENSORS", "PARAMS", "QUANT TYPES"})
			for _, l := range info.Layers {
				compStr := strings.Join(l.Components, ", ")
				if runewidth.StringWidth(compStr) > 45 {
					compStr = compStr[:42] + "..."
				}
				rows = append(rows, []string{
					l.Name,
					compStr,
					strconv.Itoa(l.TensorCount),
					format.HumanNumber(l.ParamCount),
					strings.Join(l.DataTypes, ", "),
				})
			}
			return rows
		})
	}

	// 6. GGUF Metadata / Parameters
	if (showAll || showMetadata) && len(info.Metadata) > 0 {
		tableRender("GGUF Metadata (Parameters)", func() [][]string {
			keys := make([]string, 0, len(info.Metadata))
			for k := range info.Metadata {
				keys = append(keys, k)
			}
			sort.Strings(keys)

			var rows [][]string
			for _, k := range keys {
				val := info.Metadata[k]
				valStr := formatMetadataValue(val)
				rows = append(rows, []string{"", k, valStr})
			}
			return rows
		})
	}

	// 7. Tensors Table
	if (showAll || showTensors) && len(info.Tensors) > 0 {
		header := fmt.Sprintf("Tensors (%d total)", len(info.Tensors))
		tableRender(header, func() [][]string {
			var rows [][]string
			rows = append(rows, []string{"#", "NAME", "TYPE", "SHAPE", "PARAMS"})
			for _, t := range info.Tensors {
				shapeStrs := make([]string, len(t.Shape))
				for i, s := range t.Shape {
					shapeStrs[i] = strconv.FormatUint(s, 10)
				}
				rows = append(rows, []string{
					strconv.Itoa(t.Index),
					t.Name,
					t.Type,
					fmt.Sprintf("[%s]", strings.Join(shapeStrs, ", ")),
					formatComma(t.ParamCount),
				})
			}
			return rows
		})
	}

	return nil
}

// formatMetadataValue cleans and formats arbitrary GGUF metadata values for display.
func formatMetadataValue(v any) string {
	if v == nil {
		return ""
	}

	valRef := reflect.ValueOf(v)
	if valRef.Kind() == reflect.Slice || valRef.Kind() == reflect.Array {
		length := valRef.Len()
		if length > 5 {
			var sample []string
			for i := 0; i < 3 && i < length; i++ {
				s := fmt.Sprintf("%v", valRef.Index(i).Interface())
				if len(s) > 20 {
					s = s[:17] + "..."
				}
				sample = append(sample, s)
			}
			return fmt.Sprintf("[%d items] [%s, ...]", length, strings.Join(sample, ", "))
		}
		var items []string
		for i := 0; i < length; i++ {
			items = append(items, fmt.Sprintf("%v", valRef.Index(i).Interface()))
		}
		res := fmt.Sprintf("[%s]", strings.Join(items, ", "))
		if len(res) > 80 {
			return res[:77] + "..."
		}
		return res
	}

	switch val := v.(type) {
	case bool:
		return fmt.Sprintf("%t", val)
	case string:
		if len(val) > 80 {
			return val[:77] + "..."
		}
		return val
	case float64:
		return fmt.Sprintf("%g", val)
	case int:
		return strconv.Itoa(val)
	case int64:
		return strconv.FormatInt(val, 10)
	case uint64:
		return strconv.FormatUint(val, 10)
	default:
		s := fmt.Sprintf("%v", val)
		if len(s) > 80 {
			return s[:77] + "..."
		}
		return s
	}
}

// formatComma formats large uint64 numbers with commas.
func formatComma(n uint64) string {
	in := strconv.FormatUint(n, 10)
	var out []byte
	l := len(in)
	for i, c := range []byte(in) {
		if i > 0 && (l-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, c)
	}
	return string(out)
}
