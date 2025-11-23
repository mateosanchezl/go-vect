package chunking

import "testing"

func TestFixedChunkerChunkShortText(t *testing.T) {
	chunker := FixedChunker{ChunkSize: 10}
	text := "short"

	chunks := chunker.Chunk(text)

	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0] != text {
		t.Fatalf("expected chunk to be %q, got %q", text, chunks[0])
	}
}

func TestFixedChunkerChunkSplitsText(t *testing.T) {
	chunker := FixedChunker{ChunkSize: 4}
	text := "abcdefghijkl"

	chunks := chunker.Chunk(text)

	expected := []string{"abcd", "efgh", "ijkl"}
	if len(chunks) != len(expected) {
		t.Fatalf("expected %d chunks, got %d", len(expected), len(chunks))
	}
	for i, chunk := range chunks {
		if chunk != expected[i] {
			t.Errorf("chunk %d: expected %q, got %q", i, expected[i], chunk)
		}
	}
}
