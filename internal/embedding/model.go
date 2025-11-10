package embedding

import (
	"errors"
	"fmt"
)

// Vector definition
type EmbeddingVector []float64

func NewVector(val []float64) (EmbeddingVector, error) {
	if len(val) == 0 {
		return nil, errors.New("embedding vector cannot be empty")
	}
	if len(val) != 768 {
		msg := fmt.Sprintf("embedding vector must be of length 768, got %v", len(val))
		return nil, errors.New(msg)
	}

	return EmbeddingVector(val), nil
}

// Model interface
type EmbeddingModel interface {
	Embed(chunk string) (embedding EmbeddingVector, err error)
	EmbedBatch(chunks []string) (embeddings []EmbeddingVector, err error)
}
