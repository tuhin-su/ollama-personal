package cmd

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"
)

// MemoryEntry represents a single unit of learned knowledge, user correction, or interaction.
type MemoryEntry struct {
	ID           string   `json:"id"`
	Timestamp    string   `json:"timestamp"`
	Type         string   `json:"type"`                 // "correction", "fact", "dialogue", "preference", "instruction"
	Topic        string   `json:"topic,omitempty"`      // Entity/topic key to prevent contradictions (Fault 1 fix)
	Prompt       string   `json:"prompt"`               // Context, situation, or user prompt
	Completion   string   `json:"completion"`           // Learned target, correction, or ideal response
	Source       string   `json:"source,omitempty"`
	Tags         []string `json:"tags,omitempty"`
	Words        int      `json:"words"`
	SupersededBy string   `json:"superseded_by,omitempty"` // ID of the newer memory that replaced this one (Fault 1 fix)
}

// MemoryStoreInfo summarizes the on-disk self-learning memory storage.
type MemoryStoreInfo struct {
	StorageDir   string         `json:"storage_dir"`
	ActivePath   string         `json:"active_path"`
	ArchivePath  string         `json:"archive_path"`
	ActiveSize   int64          `json:"active_size_bytes"`
	ArchiveSize  int64          `json:"archive_size_bytes"`
	ActiveCount  int            `json:"active_count"`
	ArchiveCount int            `json:"archive_count"`
	TotalCount   int            `json:"total_count"`
	ActiveWords  int            `json:"active_words"`
	ArchiveWords int            `json:"archive_words"`
	TypeCounts   map[string]int `json:"type_counts"`
	RecentActive []MemoryEntry  `json:"recent_active,omitempty"`
}

// Security sanitizer patterns to block prompt injection (Fault 4 fix).
var promptInjectionPatterns = []string{
	"ignore all previous",
	"ignore previous instructions",
	"disregard all previous",
	"disregard previous",
	"system override",
	"you are now dan",
	"developer mode enabled",
	"<|im_start|>",
	"<|im_end|>",
	"<|system|>",
	"<|user|>",
	"<|assistant|>",
}

// SanitizeMemoryInput inspects input for injection attacks or malformed strings.
func SanitizeMemoryInput(prompt, completion string) error {
	pLower := strings.ToLower(prompt)
	cLower := strings.ToLower(completion)
	for _, pattern := range promptInjectionPatterns {
		if strings.Contains(pLower, pattern) || strings.Contains(cLower, pattern) {
			return fmt.Errorf("security violation: input contains potential prompt injection pattern (%q)", pattern)
		}
	}
	if len(strings.TrimSpace(prompt)) < 3 {
		return fmt.Errorf("prompt is too short (minimum 3 characters)")
	}
	if len(strings.TrimSpace(completion)) < 2 {
		return fmt.Errorf("completion is too short (minimum 2 characters)")
	}
	return nil
}

// MemoryRAMCache provides sub-millisecond in-memory lookups without disk bottleneck (Fault 5 fix).
type MemoryRAMCache struct {
	mu          sync.RWMutex
	lastModTime time.Time
	entries     []MemoryEntry
}

var globalMemoryCache = &MemoryRAMCache{}

// InvalidateMemoryCache forces a reload on the next memory access.
func InvalidateMemoryCache() {
	globalMemoryCache.mu.Lock()
	globalMemoryCache.entries = nil
	globalMemoryCache.lastModTime = time.Time{}
	globalMemoryCache.mu.Unlock()
}

// GetActiveMemoriesCached retrieves active staged memories from RAM (reloading from disk only when modified).
func GetActiveMemoriesCached() ([]MemoryEntry, error) {
	dir, err := GetDevMemoryDir()
	if err != nil {
		return nil, err
	}
	activePath := filepath.Join(dir, "active.jsonl")

	globalMemoryCache.mu.RLock()
	stat, statErr := os.Stat(activePath)
	if statErr == nil && !stat.ModTime().After(globalMemoryCache.lastModTime) && globalMemoryCache.entries != nil {
		cached := make([]MemoryEntry, len(globalMemoryCache.entries))
		copy(cached, globalMemoryCache.entries)
		globalMemoryCache.mu.RUnlock()
		return cached, nil
	}
	globalMemoryCache.mu.RUnlock()

	// Reload into RAM
	globalMemoryCache.mu.Lock()
	defer globalMemoryCache.mu.Unlock()

	entries, err := LoadMemoryEntries(activePath)
	if err != nil {
		return nil, err
	}
	if stat != nil {
		globalMemoryCache.lastModTime = stat.ModTime()
	}
	globalMemoryCache.entries = entries

	res := make([]MemoryEntry, len(entries))
	copy(res, entries)
	return res, nil
}

// GetDevMemoryDir returns the directory where memory buffers are stored (~/.ollama/dev/memory).
func GetDevMemoryDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to determine user home directory: %w", err)
	}
	dir := filepath.Join(home, ".ollama", "dev", "memory")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("failed to create memory directory '%s': %w", dir, err)
	}
	return dir, nil
}

// GetDevAdaptersDir returns the directory where trained LoRA adapters are saved (~/.ollama/dev/adapters).
func GetDevAdaptersDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to determine user home directory: %w", err)
	}
	dir := filepath.Join(home, ".ollama", "dev", "adapters")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("failed to create adapters directory '%s': %w", dir, err)
	}
	return dir, nil
}

// LoadMemoryEntries reads all entries from a JSONL memory file.
func LoadMemoryEntries(filePath string) ([]MemoryEntry, error) {
	f, err := os.Open(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return []MemoryEntry{}, nil
		}
		return nil, err
	}
	defer f.Close()

	var entries []MemoryEntry
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), 2*1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var entry MemoryEntry
		if err := json.Unmarshal([]byte(line), &entry); err == nil {
			if entry.Words == 0 {
				entry.Words = len(strings.Fields(entry.Prompt)) + len(strings.Fields(entry.Completion))
			}
			entries = append(entries, entry)
		}
	}
	return entries, scanner.Err()
}

// SaveMemoryEntry adds a new memory, resolving contradictions by superseding older entries (Fault 1 & 4 fixes).
func SaveMemoryEntry(entry MemoryEntry) error {
	if err := SanitizeMemoryInput(entry.Prompt, entry.Completion); err != nil {
		return err
	}

	dir, err := GetDevMemoryDir()
	if err != nil {
		return err
	}

	if entry.ID == "" {
		b := make([]byte, 4)
		_, _ = rand.Read(b)
		entry.ID = fmt.Sprintf("mem_%d_%s", time.Now().Unix(), hex.EncodeToString(b))
	}
	if entry.Timestamp == "" {
		entry.Timestamp = time.Now().UTC().Format(time.RFC3339)
	}
	if entry.Type == "" {
		entry.Type = "correction"
	}
	if entry.Words == 0 {
		entry.Words = len(strings.Fields(entry.Prompt)) + len(strings.Fields(entry.Completion))
	}

	activePath := filepath.Join(dir, "active.jsonl")
	existingEntries, _ := LoadMemoryEntries(activePath)

	// Check for contradiction / superseding (Fault 1 fix)
	updated := false
	newPromptTokens := tokenizeAndFilter(entry.Prompt)

	for i := range existingEntries {
		old := &existingEntries[i]
		if old.SupersededBy != "" {
			continue
		}

		// If explicit topic matches or prompt similarity > 85%, supersede the older fact!
		topicMatch := entry.Topic != "" && strings.EqualFold(old.Topic, entry.Topic)
		similarity := computePromptSimilarity(newPromptTokens, old.Prompt)

		if topicMatch || similarity >= 0.85 {
			old.SupersededBy = entry.ID
			updated = true
		}
	}

	existingEntries = append(existingEntries, entry)

	// Write back active.jsonl atomically
	tmpPath := activePath + ".tmp"
	f, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("failed to open temp active memory file: %w", err)
	}

	for _, e := range existingEntries {
		b, _ := json.Marshal(e)
		_, _ = f.Write(append(b, '\n'))
	}
	f.Close()

	if err := os.Rename(tmpPath, activePath); err != nil {
		return fmt.Errorf("failed to commit active memory file: %w", err)
	}

	InvalidateMemoryCache()
	if updated {
		fmt.Printf("  ↻ Automatically superseded older conflicting memory with updated ground truth.\n")
	}
	return nil
}

// ArchiveActiveMemory graduates all staged active memories into archive.jsonl (local_old) and clears active.jsonl.
func ArchiveActiveMemory() (int, error) {
	dir, err := GetDevMemoryDir()
	if err != nil {
		return 0, err
	}

	activePath := filepath.Join(dir, "active.jsonl")
	archivePath := filepath.Join(dir, "archive.jsonl")

	entries, err := LoadMemoryEntries(activePath)
	if err != nil {
		return 0, err
	}
	if len(entries) == 0 {
		return 0, nil
	}

	// Filter out superseded memories before archiving
	var toArchive []MemoryEntry
	for _, e := range entries {
		if e.SupersededBy == "" {
			toArchive = append(toArchive, e)
		}
	}

	archF, err := os.OpenFile(archivePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return 0, fmt.Errorf("failed to open archive memory file: %w", err)
	}
	defer archF.Close()

	for _, e := range toArchive {
		b, _ := json.Marshal(e)
		_, _ = archF.Write(append(b, '\n'))
	}

	// Truncate active.jsonl to 0 bytes
	if err := os.Truncate(activePath, 0); err != nil {
		return 0, fmt.Errorf("failed to clear active memory file: %w", err)
	}

	InvalidateMemoryCache()
	return len(toArchive), nil
}

// ClearActiveMemory clears active.jsonl. If includeArchive is true, it also clears archive.jsonl.
func ClearActiveMemory(includeArchive bool) error {
	dir, err := GetDevMemoryDir()
	if err != nil {
		return err
	}

	activePath := filepath.Join(dir, "active.jsonl")
	_ = os.Truncate(activePath, 0)

	if includeArchive {
		archivePath := filepath.Join(dir, "archive.jsonl")
		_ = os.Truncate(archivePath, 0)
	}
	InvalidateMemoryCache()
	return nil
}

// MatchRelevantMemories searches ONLY active staged memories (Fault 2 fix: no double-learning overlap with LoRA).
// Uses strict relevance threshold (Fault 3 fix: no false matches).
func MatchRelevantMemories(userPrompt string, maxK int, minScore float64) []MemoryEntry {
	if minScore <= 0 {
		minScore = 0.60 // Strict 60% minimum relevance threshold
	}
	if maxK <= 0 {
		maxK = 3
	}

	entries, err := GetActiveMemoriesCached()
	if err != nil || len(entries) == 0 {
		return nil
	}

	type scoredMatch struct {
		entry MemoryEntry
		score float64
	}
	var matches []scoredMatch

	queryTokens := tokenizeAndFilter(userPrompt)
	if len(queryTokens) == 0 {
		return nil
	}

	for _, entry := range entries {
		// Ignore superseded entries (Fault 1 fix)
		if entry.SupersededBy != "" {
			continue
		}

		score := computeRelevanceScore(queryTokens, entry)
		if score >= minScore {
			matches = append(matches, scoredMatch{entry: entry, score: score})
		}
	}

	sort.Slice(matches, func(i, j int) bool {
		return matches[i].score > matches[j].score
	})

	if len(matches) > maxK {
		matches = matches[:maxK]
	}

	result := make([]MemoryEntry, len(matches))
	for i, m := range matches {
		result[i] = m.entry
	}
	return result
}

// FormatMemoryPrompt constructs: user_input + [memory] -> AI model.
// If no relevant memories exist, returns the original prompt unchanged (0 tokens wasted).
func FormatMemoryPrompt(userPrompt string, memories []MemoryEntry) string {
	if len(memories) == 0 {
		return userPrompt
	}

	var b strings.Builder
	b.WriteString("[Recalled Long-Term Context]\n")
	for _, m := range memories {
		cat := strings.ToUpper(m.Type)
		b.WriteString(fmt.Sprintf("• %s: %s\n", cat, m.Completion))
	}
	b.WriteString("\n[User Query]\n")
	b.WriteString(userPrompt)
	return b.String()
}

// Stop words ignored during semantic matching to prevent false positive triggers (Fault 3 fix).
var stopWords = map[string]struct{}{
	"the": {}, "a": {}, "an": {}, "and": {}, "or": {}, "in": {}, "on": {}, "at": {},
	"to": {}, "is": {}, "are": {}, "was": {}, "were": {}, "of": {}, "for": {}, "with": {},
	"by": {}, "it": {}, "this": {}, "that": {}, "what": {}, "how": {}, "where": {}, "why": {},
	"can": {}, "do": {}, "does": {}, "we": {}, "you": {}, "i": {}, "my": {}, "our": {},
	"me": {}, "your": {}, "be": {}, "as": {}, "from": {}, "so": {},
}

func tokenizeAndFilter(text string) []string {
	clean := strings.ToLower(text)
	var tokens []string
	words := strings.FieldsFunc(clean, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_' || r == '-')
	})
	for _, w := range words {
		if len(w) <= 1 {
			continue
		}
		if _, isStop := stopWords[w]; isStop {
			continue
		}
		tokens = append(tokens, w)
	}
	return tokens
}

func computeRelevanceScore(queryTokens []string, entry MemoryEntry) float64 {
	if len(queryTokens) == 0 {
		return 0
	}
	entryTokens := tokenizeAndFilter(entry.Prompt + " " + strings.Join(entry.Tags, " ") + " " + entry.Topic)
	if len(entryTokens) == 0 {
		return 0
	}

	entryMap := make(map[string]bool)
	for _, t := range entryTokens {
		entryMap[t] = true
	}

	matchedCount := 0
	for _, q := range queryTokens {
		if entryMap[q] {
			matchedCount++
		}
	}

	if matchedCount == 0 {
		return 0
	}

	// Recall over user query terms: how much of user's intent is addressed by this memory
	score := float64(matchedCount) / float64(len(queryTokens))

	// Bonus if query mentions topic
	if entry.Topic != "" && strings.Contains(strings.ToLower(strings.Join(queryTokens, " ")), strings.ToLower(entry.Topic)) {
		score += 0.20
	}

	if score > 1.0 {
		score = 1.0
	}
	return score
}

func computePromptSimilarity(tokens1 []string, prompt2 string) float64 {
	tokens2 := tokenizeAndFilter(prompt2)
	if len(tokens1) == 0 || len(tokens2) == 0 {
		return 0
	}
	m2 := make(map[string]bool)
	for _, t := range tokens2 {
		m2[t] = true
	}
	intersection := 0
	for _, t := range tokens1 {
		if m2[t] {
			intersection++
		}
	}
	union := len(tokens1) + len(tokens2) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

// GetMemoryStoreInfo returns detailed metrics on memory storage size, entries, and categories.
func GetMemoryStoreInfo() (*MemoryStoreInfo, error) {
	dir, err := GetDevMemoryDir()
	if err != nil {
		return nil, err
	}

	activePath := filepath.Join(dir, "active.jsonl")
	archivePath := filepath.Join(dir, "archive.jsonl")

	activeEntries, _ := LoadMemoryEntries(activePath)
	archiveEntries, _ := LoadMemoryEntries(archivePath)

	var activeSize int64
	if stat, err := os.Stat(activePath); err == nil {
		activeSize = stat.Size()
	}

	var archiveSize int64
	if stat, err := os.Stat(archivePath); err == nil {
		archiveSize = stat.Size()
	}

	typeCounts := make(map[string]int)
	activeWords := 0
	validActive := 0
	for _, e := range activeEntries {
		if e.SupersededBy == "" {
			typeCounts[e.Type]++
			activeWords += e.Words
			validActive++
		}
	}

	archiveWords := 0
	for _, e := range archiveEntries {
		typeCounts[e.Type]++
		archiveWords += e.Words
	}

	var recent []MemoryEntry
	for _, e := range activeEntries {
		if e.SupersededBy == "" {
			recent = append(recent, e)
		}
	}
	if len(recent) > 10 {
		recent = recent[len(recent)-10:]
	}

	return &MemoryStoreInfo{
		StorageDir:   dir,
		ActivePath:   activePath,
		ArchivePath:  archivePath,
		ActiveSize:   activeSize,
		ArchiveSize:  archiveSize,
		ActiveCount:  validActive,
		ArchiveCount: len(archiveEntries),
		TotalCount:   validActive + len(archiveEntries),
		ActiveWords:  activeWords,
		ArchiveWords: archiveWords,
		TypeCounts:   typeCounts,
		RecentActive: recent,
	}, nil
}

// NewDevMemoryCommand creates the 'dev memory' command and subcommands.
func NewDevMemoryCommand() *cobra.Command {
	memoryCmd := &cobra.Command{
		Use:   "memory",
		Short: "Inspect, manage, and view self-learning memory data (size, storage, entries)",
		RunE:  DevMemoryStatusHandler,
	}

	memoryCmd.Flags().BoolP("json", "j", false, "Output memory metrics and entries in JSON format")
	memoryCmd.Flags().BoolP("all", "a", false, "Display all active memory entries in the table")

	// Subcommand: add
	addCmd := &cobra.Command{
		Use:   "add",
		Short: "Add a learned insight, fact, or correction to the staged active memory buffer",
		RunE:  DevMemoryAddHandler,
	}
	addCmd.Flags().StringP("prompt", "p", "", "User prompt or situation context (required)")
	addCmd.Flags().StringP("completion", "c", "", "Learned target answer or correction (required)")
	addCmd.Flags().StringP("type", "t", "correction", "Memory category (correction, fact, dialogue, instruction, preference)")
	addCmd.Flags().String("topic", "", "Optional entity/topic key (used to supersede older conflicting facts)")
	addCmd.Flags().String("tags", "", "Comma-separated tags (e.g. 'coding,go,logic')")
	_ = addCmd.MarkFlagRequired("prompt")
	_ = addCmd.MarkFlagRequired("completion")

	// Subcommand: query / recall test
	queryCmd := &cobra.Command{
		Use:   "recall PROMPT",
		Short: "Simulate live memory retrieval for a prompt (checks what memories would be injected)",
		Args:  cobra.ExactArgs(1),
		RunE:  DevMemoryRecallHandler,
	}
	queryCmd.Flags().Float64("threshold", 0.60, "Minimum similarity score threshold (0.0 to 1.0)")
	queryCmd.Flags().IntP("max", "k", 3, "Maximum memories to retrieve")

	// Subcommand: clear
	clearCmd := &cobra.Command{
		Use:   "clear",
		Short: "Clear the staged active memory buffer",
		RunE:  DevMemoryClearHandler,
	}
	clearCmd.Flags().BoolP("all", "a", false, "Also clear the historical trained archive (local_old)")

	// Subcommand: export
	exportCmd := &cobra.Command{
		Use:   "export [FILE]",
		Short: "Export memory dataset to a JSONL training file",
		Args:  cobra.MaximumNArgs(1),
		RunE:  DevMemoryExportHandler,
	}
	exportCmd.Flags().Bool("include-archive", true, "Include historical local_old archive memories with new_local staged memories")

	memoryCmd.AddCommand(addCmd, queryCmd, clearCmd, exportCmd)
	return memoryCmd
}

// DevMemoryStatusHandler renders complete memory storage details and recent entries.
func DevMemoryStatusHandler(cmd *cobra.Command, args []string) error {
	info, err := GetMemoryStoreInfo()
	if err != nil {
		return err
	}

	asJSON, _ := cmd.Flags().GetBool("json")
	if asJSON {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		return encoder.Encode(info)
	}

	showAll, _ := cmd.Flags().GetBool("all")
	return renderMemoryStatusText(os.Stdout, info, showAll)
}

// DevMemoryAddHandler adds a memory entry with contradiction and security checks.
func DevMemoryAddHandler(cmd *cobra.Command, args []string) error {
	prompt, _ := cmd.Flags().GetString("prompt")
	completion, _ := cmd.Flags().GetString("completion")
	memType, _ := cmd.Flags().GetString("type")
	topic, _ := cmd.Flags().GetString("topic")
	tagsStr, _ := cmd.Flags().GetString("tags")

	var tags []string
	if tagsStr != "" {
		for _, t := range strings.Split(tagsStr, ",") {
			trimmed := strings.TrimSpace(t)
			if trimmed != "" {
				tags = append(tags, trimmed)
			}
		}
	}

	entry := MemoryEntry{
		Type:       memType,
		Topic:      topic,
		Prompt:     prompt,
		Completion: completion,
		Source:     "user_manual",
		Tags:       tags,
	}

	if err := SaveMemoryEntry(entry); err != nil {
		return fmt.Errorf("failed to save memory: %w", err)
	}

	info, _ := GetMemoryStoreInfo()
	fmt.Printf("✓ Added memory [%s] to staged active buffer (security & contradiction checks passed)\n", memType)
	fmt.Printf("  Prompt:     %s\n", truncatePreview(prompt, 60))
	fmt.Printf("  Completion: %s\n", truncatePreview(completion, 60))
	if topic != "" {
		fmt.Printf("  Topic:      %s\n", topic)
	}
	if info != nil {
		fmt.Printf("  Active Staged Buffer: %d entries (ready for live recall and LoRA train)\n", info.ActiveCount)
	}
	return nil
}

// DevMemoryRecallHandler tests memory recall for a query to verify what gets injected.
func DevMemoryRecallHandler(cmd *cobra.Command, args []string) error {
	query := args[0]
	threshold, _ := cmd.Flags().GetFloat64("threshold")
	maxK, _ := cmd.Flags().GetInt("max")

	matches := MatchRelevantMemories(query, maxK, threshold)

	fmt.Printf("  Memory Recall Test for Query: %q\n", query)
	fmt.Printf("  Threshold: %.2f | Max Items: %d\n\n", threshold, maxK)

	if len(matches) == 0 {
		fmt.Printf("  → Zero memories matched threshold (0 extra tokens injected into AI prompt).\n")
		return nil
	}

	fmt.Printf("  → %d Relevant Memories Recalled (Ready for Context Injection):\n", len(matches))
	for i, m := range matches {
		score := computeRelevanceScore(tokenizeAndFilter(query), m)
		fmt.Printf("    [%d] Score: %.2f | Type: %s | ID: %s\n", i+1, score, m.Type, m.ID)
		fmt.Printf("        Recall Truth: %s\n", m.Completion)
	}

	fmt.Printf("\n  Simulated Injected Prompt:\n")
	fmt.Printf("  ────────────────────────────────────────────────────────────\n")
	formatted := FormatMemoryPrompt(query, matches)
	fmt.Println(formatted)
	fmt.Printf("  ────────────────────────────────────────────────────────────\n")
	return nil
}

// DevMemoryClearHandler clears memory buffers.
func DevMemoryClearHandler(cmd *cobra.Command, args []string) error {
	includeArchive, _ := cmd.Flags().GetBool("all")
	if err := ClearActiveMemory(includeArchive); err != nil {
		return fmt.Errorf("failed to clear memory: %w", err)
	}

	if includeArchive {
		fmt.Println("✓ Cleared all self-learning memory (both staged active buffer and historical archive).")
	} else {
		fmt.Println("✓ Cleared staged active memory buffer (historical archive preserved).")
	}
	return nil
}

// DevMemoryExportHandler exports dataset to JSONL file.
func DevMemoryExportHandler(cmd *cobra.Command, args []string) error {
	includeArchive, _ := cmd.Flags().GetBool("include-archive")
	dir, err := GetDevMemoryDir()
	if err != nil {
		return err
	}

	outPath := "ollama_memory_dataset.jsonl"
	if len(args) > 0 {
		outPath = args[0]
	}

	activeEntries, _ := LoadMemoryEntries(filepath.Join(dir, "active.jsonl"))
	var allEntries []MemoryEntry
	if includeArchive {
		archEntries, _ := LoadMemoryEntries(filepath.Join(dir, "archive.jsonl"))
		allEntries = append(allEntries, archEntries...)
	}
	for _, e := range activeEntries {
		if e.SupersededBy == "" {
			allEntries = append(allEntries, e)
		}
	}

	if len(allEntries) == 0 {
		return fmt.Errorf("no memory entries found to export")
	}

	outFile, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("failed to create export file '%s': %w", outPath, err)
	}
	defer outFile.Close()

	for _, e := range allEntries {
		row := map[string]any{
			"prompt":     e.Prompt,
			"completion": e.Completion,
			"type":       e.Type,
			"topic":      e.Topic,
			"timestamp":  e.Timestamp,
		}
		b, _ := json.Marshal(row)
		_, _ = outFile.Write(append(b, '\n'))
	}

	fmt.Printf("✓ Exported %d memory records (mode: %s) to '%s'\n", len(allEntries), func() string {
		if includeArchive {
			return "local_old + new_local"
		}
		return "new_local only"
	}(), outPath)
	return nil
}

// renderMemoryStatusText displays clean terminal formatting for memory inspection.
func renderMemoryStatusText(w io.Writer, info *MemoryStoreInfo, showAll bool) error {
	fmt.Fprintf(w, "  Self-Learning Memory Overview\n")
	fmt.Fprintf(w, "    Storage Directory      %s\n", info.StorageDir)
	fmt.Fprintf(w, "    Active Buffer File     %s\n", info.ActivePath)
	fmt.Fprintf(w, "    Historical Archive     %s\n", info.ArchivePath)
	fmt.Fprintln(w)

	fmt.Fprintf(w, "  Memory Capacity & Data Sizes\n")
	fmt.Fprintf(w, "    Staged Active Buffer   %s (%d entries, ~%s words) [Ready for next LoRA train]\n",
		formatBytes(info.ActiveSize), info.ActiveCount, formatNumber(info.ActiveWords))
	fmt.Fprintf(w, "    Historical Archive     %s (%d entries, ~%s words) [From prior LoRA sessions]\n",
		formatBytes(info.ArchiveSize), info.ArchiveCount, formatNumber(info.ArchiveWords))
	fmt.Fprintf(w, "    Total Cumulative       %s (%d entries, ~%s words) [Mode: local_old + new_local]\n",
		formatBytes(info.ActiveSize+info.ArchiveSize), info.TotalCount, formatNumber(info.ActiveWords+info.ArchiveWords))
	fmt.Fprintln(w)

	if len(info.TypeCounts) > 0 {
		fmt.Fprintf(w, "  Knowledge Categories Breakdown\n")
		for cat, count := range info.TypeCounts {
			fmt.Fprintf(w, "    • %-16s %d entries\n", cat, count)
		}
		fmt.Fprintln(w)
	}

	activeEntries, _ := LoadMemoryEntries(info.ActivePath)
	var validActive []MemoryEntry
	for _, e := range activeEntries {
		if e.SupersededBy == "" {
			validActive = append(validActive, e)
		}
	}

	if len(validActive) > 0 {
		displayList := validActive
		if !showAll && len(displayList) > 8 {
			displayList = displayList[len(displayList)-8:]
		}

		fmt.Fprintf(w, "  Staged Active Memories (%d displayed of %d total active)\n", len(displayList), len(validActive))
		table := tablewriter.NewWriter(w)
		table.SetHeader([]string{"ID", "Type", "Topic", "Prompt Preview", "Learned Target / Correction", "Words"})
		table.SetBorders(tablewriter.Border{Left: false, Top: false, Right: false, Bottom: false})
		table.SetCenterSeparator("")
		table.SetColumnSeparator("  ")
		table.SetAutoWrapText(false)

		for _, e := range displayList {
			table.Append([]string{
				truncatePreview(e.ID, 16),
				e.Type,
				truncatePreview(e.Topic, 12),
				truncatePreview(e.Prompt, 32),
				truncatePreview(e.Completion, 32),
				fmt.Sprintf("%d", e.Words),
			})
		}
		table.Render()
		fmt.Fprintln(w)

		fmt.Fprintf(w, "  Next Steps:\n")
		fmt.Fprintf(w, "    Test live recall:         ollama dev memory recall \"Your prompt\"\n")
		fmt.Fprintf(w, "    Run LoRA fine-tuning:     ollama dev train <model>\n")
	} else {
		fmt.Fprintf(w, "  Active Buffer is currently empty.\n")
		fmt.Fprintf(w, "    Add memories manually:    ollama dev memory add -p \"Context\" -c \"Correction\"\n")
		fmt.Fprintf(w, "    Or interact with model:   Memory accumulates automatically during sessions.\n")
	}

	return nil
}

func truncatePreview(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	if len(s) > maxLen {
		return s[:maxLen-3] + "..."
	}
	return s
}

func formatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

func formatNumber(n int) string {
	in := fmt.Sprintf("%d", n)
	var out []rune
	for i, r := range in {
		if i > 0 && (len(in)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, r)
	}
	return string(out)
}
