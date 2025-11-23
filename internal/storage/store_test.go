package storage

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mateosanchezl/go-vect/internal/embedding"
)

const embeddingSize = 384

func setupTempDB(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	vectorPath := filepath.Join(dir, "vectors.bin")
	metaPath := filepath.Join(dir, "metadata.jsonl")
	t.Setenv("VECTOR_DB_PATH", vectorPath)
	t.Setenv("METADATA_DB_PATH", metaPath)
	return vectorPath, metaPath
}

func newSparseVector(entries map[int]float32) embedding.EmbeddingVector {
	v := make(embedding.EmbeddingVector, embeddingSize)
	for idx, val := range entries {
		v[idx] = val
	}
	return v
}

func TestVectorToByteSliceRoundTrip(t *testing.T) {
	v := embedding.EmbeddingVector{1.5, -2.25, 0.5}

	b := vectorToByteSlice(v)
	if len(b) != len(v)*4 {
		t.Fatalf("expected %d bytes, got %d", len(v)*4, len(b))
	}

	for i := range v {
		back := mathFromBytes(b[i*4 : (i+1)*4])
		if back != v[i] {
			t.Fatalf("index %d: expected %v, got %v", i, v[i], back)
		}
	}
}

func TestStoreEmbeddingWritesDataAndMetadata(t *testing.T) {
	vectorPath, metaPath := setupTempDB(t)
	vec := newSparseVector(map[int]float32{0: 3, 1: 4})

	if err := StoreEmbedding(vec, "first"); err != nil {
		t.Fatalf("store embedding (first): %v", err)
	}
	if err := StoreEmbedding(vec, "second"); err != nil {
		t.Fatalf("store embedding (second): %v", err)
	}

	data, err := os.ReadFile(vectorPath)
	if err != nil {
		t.Fatalf("reading vector db: %v", err)
	}
	expectedBytes := len(vec) * 4 * 2
	if len(data) != expectedBytes {
		t.Fatalf("expected %d bytes stored, got %d", expectedBytes, len(data))
	}

	firstVal := mathFromBytes(data[0:4])
	secondVal := mathFromBytes(data[4:8])
	if diff := abs(firstVal - 0.6); diff > 1e-5 {
		t.Fatalf("expected normalised first value 0.6, got %v", firstVal)
	}
	if diff := abs(secondVal - 0.8); diff > 1e-5 {
		t.Fatalf("expected normalised second value 0.8, got %v", secondVal)
	}

	mdBytes, err := os.ReadFile(metaPath)
	if err != nil {
		t.Fatalf("reading metadata: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(mdBytes)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 metadata lines, got %d", len(lines))
	}

	var first, second EmbeddingMetaData
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("unmarshal first metadata: %v", err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &second); err != nil {
		t.Fatalf("unmarshal second metadata: %v", err)
	}

	vecBytes := len(vec) * 4
	if first.Offset != vecBytes {
		t.Fatalf("expected first offset %d, got %d", vecBytes, first.Offset)
	}
	if second.Offset != vecBytes*2 {
		t.Fatalf("expected second offset %d, got %d", vecBytes*2, second.Offset)
	}
}

func TestCalculateOffsetUsesMetadata(t *testing.T) {
	_, metaPath := setupTempDB(t)

	entry := EmbeddingMetaData{Offset: 12, Text: "existing"}
	b, err := json.Marshal(entry)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(metaPath, append(b, '\n'), 0o644); err != nil {
		t.Fatalf("write metadata: %v", err)
	}

	vec := embedding.EmbeddingVector{1, 2, 3}

	offset, err := calculateOffset(vec)
	if err != nil {
		t.Fatalf("calculate offset: %v", err)
	}

	expected := entry.Offset + len(vec)*4
	if offset != expected {
		t.Fatalf("expected offset %d, got %d", expected, offset)
	}
}

func TestGetLastOffsetMissingFile(t *testing.T) {
	_, metaPath := setupTempDB(t)
	if _, err := os.Stat(metaPath); !os.IsNotExist(err) {
		t.Fatalf("expected metadata file to not exist initially")
	}

	offset, err := getLastOffset()
	if err != nil {
		t.Fatalf("getLastOffset: %v", err)
	}
	if offset != 0 {
		t.Fatalf("expected offset 0 for missing metadata, got %d", offset)
	}
}

func TestClearData(t *testing.T) {
	vectorPath, metaPath := setupTempDB(t)

	if err := os.WriteFile(vectorPath, []byte("vector-data"), 0o644); err != nil {
		t.Fatalf("write vector: %v", err)
	}
	if err := os.WriteFile(metaPath, []byte("meta-data"), 0o644); err != nil {
		t.Fatalf("write metadata: %v", err)
	}

	if err := ClearData(); err != nil {
		t.Fatalf("ClearData: %v", err)
	}

	for _, path := range []string{vectorPath, metaPath} {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("stat %s: %v", path, err)
		}
		if info.Size() != 0 {
			t.Fatalf("expected %s to be truncated, size %d", path, info.Size())
		}
	}
}

func mathFromBytes(b []byte) float32 {
	return mathFloat(binary.LittleEndian.Uint32(b))
}

func mathFloat(bits uint32) float32 {
	return math.Float32frombits(bits)
}

func abs(f float32) float32 {
	if f < 0 {
		return -f
	}
	return f
}
