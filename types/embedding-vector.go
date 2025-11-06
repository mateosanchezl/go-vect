package types

import "errors"

type EmbeddingVector []float64

func NewVector(val []float64) (EmbeddingVector, error) {
	if len(val) == 0 {
		return nil, errors.New("embedding vector cannot be empty")
	}
	if len(val) != 768 {
		return nil, errors.New("embedding vector must be of length 768")
	}

	return EmbeddingVector(val), nil
}
