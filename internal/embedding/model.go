package embedding

import (
	"fmt"
)

// Vector definition
type EmbeddingVector []float32

func NewVector(val []float32, expectedLength int) (EmbeddingVector, error) {
	if len(val) == 0 || len(val) != expectedLength {
		return nil, fmt.Errorf("embedding vector must be of length %d", expectedLength)
	}

	return EmbeddingVector(val), nil
}

// Model interface
type EmbeddingModel interface {
	Embed(chunk string) (embedding EmbeddingVector, err error)
	EmbedBatch(chunks []string) (embeddings []EmbeddingVector, err error)
}
