package search

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mateosanchezl/go-vect/internal/embedding"
	"github.com/mateosanchezl/go-vect/internal/storage"
)

const testVectorLength = 384

func setupSearchEnv(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	vectorPath := filepath.Join(dir, "vectors.bin")
	metaPath := filepath.Join(dir, "metadata.jsonl")
	t.Setenv("VECTOR_DB_PATH", vectorPath)
	t.Setenv("METADATA_DB_PATH", metaPath)
	return vectorPath, metaPath
}

func basisVector(idx int, value float32) embedding.EmbeddingVector {
	v := make(embedding.EmbeddingVector, testVectorLength)
	v[idx] = value
	return v
}

func TestReadMetadataCreatesFileWhenMissing(t *testing.T) {
	_, metaPath := setupSearchEnv(t)

	lines, err := readMetadata()
	if err != nil {
		t.Fatalf("readMetadata: %v", err)
	}
	if len(lines) != 0 {
		t.Fatalf("expected no metadata lines, got %d", len(lines))
	}

	if _, err := os.Stat(metaPath); err != nil {
		t.Fatalf("metadata file should exist after readMetadata: %v", err)
	}
}

func TestReadVectorsReturnsStoredEmbeddings(t *testing.T) {
	_, _ = setupSearchEnv(t)

	vec := basisVector(0, 1)
	if err := storage.StoreEmbedding(vec, "doc"); err != nil {
		t.Fatalf("StoreEmbedding: %v", err)
	}

	evs, err := readVectors()
	if err != nil {
		t.Fatalf("readVectors: %v", err)
	}
	if len(evs) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(evs))
	}
	if len(evs[0]) != testVectorLength {
		t.Fatalf("expected embedding length %d, got %d", testVectorLength, len(evs[0]))
	}
}

type fakeModel struct {
	vector embedding.EmbeddingVector
}

func (f *fakeModel) Embed(chunk string) (embedding.EmbeddingVector, error) {
	// Return a copy to avoid accidental mutation.
	out := make(embedding.EmbeddingVector, len(f.vector))
	copy(out, f.vector)
	return out, nil
}

func (f *fakeModel) EmbedBatch(chunks []string) ([]embedding.EmbeddingVector, error) {
	return nil, nil
}

func TestSearchTopKSimilarReturnsSortedResults(t *testing.T) {
	_, metaPath := setupSearchEnv(t)

	docs := []struct {
		vector embedding.EmbeddingVector
		text   string
	}{
		{basisVector(0, 1), "doc-one"},
		{basisVector(1, 1), "doc-two"},
		{basisVector(2, 1), "doc-three"},
	}

	for _, doc := range docs {
		if err := storage.StoreEmbedding(doc.vector, doc.text); err != nil {
			t.Fatalf("StoreEmbedding %s: %v", doc.text, err)
		}
	}

	queryVec := basisVector(1, 1)
	queryVec[0] = 0.5
	model := &fakeModel{vector: queryVec}

	results, err := SearchTopKSimilar("q", 2, model)
	if err != nil {
		t.Fatalf("SearchTopKSimilar: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	if results[0].Text != "doc-two" {
		t.Fatalf("expected top result doc-two, got %s", results[0].Text)
	}
	if results[1].Text != "doc-one" {
		t.Fatalf("expected second result doc-one, got %s", results[1].Text)
	}

	// Ensure metadata remains valid JSON.
	data, err := os.ReadFile(metaPath)
	if err != nil {
		t.Fatalf("read metadata: %v", err)
	}
	lines := splitTrim(string(data))
	if len(lines) != len(docs) {
		t.Fatalf("expected %d metadata lines, got %d", len(docs), len(lines))
	}
	for _, line := range lines {
		var md storage.EmbeddingMetaData
		if err := json.Unmarshal([]byte(line), &md); err != nil {
			t.Fatalf("metadata line %q is not valid JSON: %v", line, err)
		}
		if md.Text == "" || md.Offset == 0 {
			t.Fatalf("metadata %+v missing fields", md)
		}
	}
}

func splitTrim(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return []string{}
	}
	return strings.Split(s, "\n")
}
