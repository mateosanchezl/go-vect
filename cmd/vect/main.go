package main

import (
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/mateosanchezl/go-vect/internal/chunking"
	"github.com/mateosanchezl/go-vect/internal/config"
	"github.com/mateosanchezl/go-vect/internal/embedding"
	"github.com/mateosanchezl/go-vect/internal/search"
	"github.com/mateosanchezl/go-vect/internal/storage"
)

type (
	stage int
	op    int
)

const (
	stageMenu stage = iota
	stageInput
)

const (
	opNone op = iota
	opEmbedText
	opEmbedFile
	opSearch
	opDeleteData
)

type menuItem struct {
	title       string
	description string
	action      op
}

type model struct {
	chunker  chunking.Chunker
	embedder embedding.EmbeddingModel

	menu       []menuItem
	menuIndex  int
	stage      stage
	activeOp   op
	inputLabel string
	input      textinput.Model

	loading        bool
	loadingMessage string

	statusLines []string
	results     []search.TopKSearchResult
	err         error
}

type opResultMsg struct {
	operation op
	lines     []string
	results   []search.TopKSearchResult
}

type opErrorMsg struct {
	operation op
	err       error
}

func main() {
	if err := config.Load(); err != nil {
		log.Fatal("failed to load config:", err)
	}

	initial := newModel()
	if _, err := tea.NewProgram(initial).Run(); err != nil {
		log.Fatal("failed to start TUI:", err)
	}
}

func newModel() model {
	ti := textinput.New()
	ti.Prompt = "> "
	ti.CharLimit = 0
	ti.Placeholder = ""
	ti.Blur()

	return model{
		chunker:  &chunking.DelimiterChunker{Delimiter: "."},
		embedder: &embedding.MiniLM{},
		menu: []menuItem{
			{title: "Embed Text", description: "Chunk, embed, and store text", action: opEmbedText},
			{title: "Embed File", description: "Read a file, chunk it, and store embeddings", action: opEmbedFile},
			{title: "Search", description: "Search stored embeddings by text query", action: opSearch},
			{title: "Delete Data", description: "Clear all stored embeddings and metadata", action: opDeleteData},
		},
		stage: stageMenu,
		input: ti,
	}
}

func (m model) Init() tea.Cmd {
	return nil
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			return m, tea.Quit
		}

		if m.loading {
			return m, nil
		}

		if m.stage == stageInput {
			var cmd tea.Cmd
			m.input, cmd = m.input.Update(msg)

			if msg.Type == tea.KeyEnter {
				val := strings.TrimSpace(m.input.Value())
				if val == "" {
					m.err = errors.New("input cannot be empty")
					return m, cmd
				}

				execCmd, loadingMessage, err := m.commandForInput(val)
				if err != nil {
					m.err = err
					return m, cmd
				}

				m.prepareForExecution(loadingMessage)
				return m, execCmd
			}

			return m, cmd
		}

		switch msg.String() {
		case "up", "k":
			if m.menuIndex > 0 {
				m.menuIndex--
			}
		case "down", "j":
			if m.menuIndex < len(m.menu)-1 {
				m.menuIndex++
			}
		case "enter":
			return m.handleMenuSelection()
		}

	case opResultMsg:
		m.loading = false
		m.loadingMessage = ""
		m.statusLines = msg.lines
		m.err = nil
		m.results = nil
		if len(msg.results) > 0 {
			m.results = msg.results
		}
		m.activeOp = opNone

	case opErrorMsg:
		m.loading = false
		m.loadingMessage = ""
		m.err = msg.err
		m.activeOp = opNone
	}

	return m, nil
}

func (m model) View() string {
	var b strings.Builder

	b.WriteString("go-vect\n")
	b.WriteString("────────\n")
	b.WriteString("Use ↑/↓ or j/k to navigate, Enter to run, q to quit.\n\n")

	for i, item := range m.menu {
		cursor := " "
		if i == m.menuIndex {
			cursor = ">"
		}
		b.WriteString(fmt.Sprintf("%s %s — %s\n", cursor, item.title, item.description))
	}

	if m.stage == stageInput {
		b.WriteString("\n")
		b.WriteString(fmt.Sprintf("%s\n", m.inputLabel))
		b.WriteString(m.input.View())
		b.WriteString("\n")
	}

	if m.loading {
		b.WriteString("\n")
		if m.loadingMessage != "" {
			b.WriteString(fmt.Sprintf("%s\n", m.loadingMessage))
		} else {
			b.WriteString("Working...\n")
		}
	}

	if m.err != nil {
		b.WriteString("\n")
		b.WriteString(fmt.Sprintf("Error: %v\n", m.err))
	}

	if len(m.statusLines) > 0 {
		b.WriteString("\nStatus:\n")
		for _, line := range m.statusLines {
			b.WriteString(fmt.Sprintf("• %s\n", line))
		}
	}

	if len(m.results) > 0 {
		b.WriteString("\nSearch Results:\n")
		for i, r := range m.results {
			b.WriteString(fmt.Sprintf("%d) cos=%.4f\n", i+1, r.CosSim))
			b.WriteString(fmt.Sprintf("   %s\n", r.Text))
		}
	}

	return b.String()
}

func (m *model) handleMenuSelection() (tea.Model, tea.Cmd) {
	if len(m.menu) == 0 {
		return m, nil
	}

	item := m.menu[m.menuIndex]
	m.err = nil
	m.results = nil
	m.statusLines = nil

	switch item.action {
	case opEmbedText:
		m.setInputMode("Enter text to embed:", "Paste or type text…", opEmbedText)
	case opEmbedFile:
		m.setInputMode("Enter file path to embed:", "/path/to/file.txt", opEmbedFile)
	case opSearch:
		m.setInputMode("Enter search query:", "What would you like to find?", opSearch)
	case opDeleteData:
		m.loading = true
		m.loadingMessage = "Deleting stored vectors…"
		m.activeOp = opDeleteData
		return m, deleteDataCmd()
	}

	return m, nil
}

func (m *model) setInputMode(label, placeholder string, action op) {
	m.stage = stageInput
	m.inputLabel = label
	m.input.Placeholder = placeholder
	m.input.SetValue("")
	m.input.Focus()
	m.activeOp = action
}

func (m *model) prepareForExecution(message string) {
	m.stage = stageMenu
	m.input.Blur()
	m.input.SetValue("")
	m.loading = true
	m.loadingMessage = message
	m.statusLines = nil
	m.results = nil
	m.err = nil
}

func (m model) commandForInput(value string) (tea.Cmd, string, error) {
	switch m.activeOp {
	case opEmbedText:
		return embedTextCmd(m.chunker, m.embedder, value), "Embedding text…", nil
	case opEmbedFile:
		return embedFileCmd(m.chunker, m.embedder, value), "Embedding file…", nil
	case opSearch:
		return searchCmd(m.embedder, value), "Searching…", nil
	default:
		return nil, "", errors.New("no action selected")
	}
}

func embedTextCmd(chunker chunking.Chunker, embedder embedding.EmbeddingModel, text string) tea.Cmd {
	return func() tea.Msg {
		lines, err := runEmbedding(chunker, embedder, text)
		if err != nil {
			return opErrorMsg{operation: opEmbedText, err: err}
		}
		return opResultMsg{operation: opEmbedText, lines: lines}
	}
}

func embedFileCmd(chunker chunking.Chunker, embedder embedding.EmbeddingModel, path string) tea.Cmd {
	return func() tea.Msg {
		cleanPath := strings.TrimSpace(path)
		if cleanPath == "" {
			return opErrorMsg{operation: opEmbedFile, err: errors.New("file path cannot be empty")}
		}

		data, err := os.ReadFile(cleanPath)
		if err != nil {
			return opErrorMsg{operation: opEmbedFile, err: fmt.Errorf("failed to read file: %w", err)}
		}

		lines, err := runEmbedding(chunker, embedder, string(data))
		if err != nil {
			return opErrorMsg{operation: opEmbedFile, err: err}
		}
		lines = append([]string{fmt.Sprintf("Source file: %s", cleanPath)}, lines...)
		return opResultMsg{operation: opEmbedFile, lines: lines}
	}
}

func searchCmd(embedder embedding.EmbeddingModel, query string) tea.Cmd {
	return func() tea.Msg {
		q := strings.TrimSpace(query)
		if q == "" {
			return opErrorMsg{operation: opSearch, err: errors.New("query cannot be empty")}
		}

		results, err := search.SearchTopKSimilar(q, 10, embedder)
		if err != nil {
			return opErrorMsg{operation: opSearch, err: fmt.Errorf("failed to search: %w", err)}
		}

		lines := []string{fmt.Sprintf("Retrieved %d results for query %q", len(results), q)}
		return opResultMsg{operation: opSearch, lines: lines, results: results}
	}
}

func deleteDataCmd() tea.Cmd {
	return func() tea.Msg {
		if err := storage.ClearData(); err != nil {
			return opErrorMsg{operation: opDeleteData, err: fmt.Errorf("failed to clear data: %w", err)}
		}
		return opResultMsg{operation: opDeleteData, lines: []string{"Successfully deleted stored vectors."}}
	}
}

func runEmbedding(chunker chunking.Chunker, embedder embedding.EmbeddingModel, text string) ([]string, error) {
	clean := strings.TrimSpace(text)
	if clean == "" {
		return nil, errors.New("no text provided to embed")
	}

	chunks := chunker.Chunk(clean)
	if len(chunks) == 0 {
		return nil, errors.New("chunker produced no chunks")
	}

	start := time.Now()
	embeddings, err := embedder.EmbedBatch(chunks)
	if err != nil {
		return nil, fmt.Errorf("failed to embed batch: %w", err)
	}
	embedElapsed := time.Since(start)

	lines := []string{fmt.Sprintf("Embedded %d chunks in %s", len(chunks), embedElapsed)}
	for i, e := range embeddings {
		lines = append(lines, fmt.Sprintf("Chunk %d embedding length: %d", i+1, len(e)))
	}

	storeStart := time.Now()
	for i, e := range embeddings {
		if err := storage.StoreEmbedding(e, chunks[i]); err != nil {
			return nil, fmt.Errorf("failed to store embedding: %w", err)
		}
	}
	storeElapsed := time.Since(storeStart)
	lines = append(lines, fmt.Sprintf("Stored embeddings in %s", storeElapsed))

	return lines, nil
}
