package embedding

import (
	"errors"
	"fmt"
	"math"
)

// Vector definition
type EmbeddingVector []float32

func NewVector(val []float32, expectedLength int) (EmbeddingVector, error) {
	if len(val) == 0 || len(val) != expectedLength {
		return nil, fmt.Errorf("embedding vector must be of length %d", expectedLength)
	}

	return EmbeddingVector(val), nil
}

// Vector ops
func (v EmbeddingVector) Dot(u EmbeddingVector) (uv float32, err error) {
	if len(v) != len(u) {
		return 0, errors.New("dot: vectors are not of same size")
	}

	var t float32
	for i := range len(u) {
		t += u[i] * (v[i])
	}
	return t, nil
}

func (v EmbeddingVector) Mag() (m float32) {
	m = 0
	for i := range len(v) {
		m += v[i] * v[i]
	}
	return float32(math.Sqrt(float64(m)))
}

// Calculate cosine similarity between two normalised vectors (just dot)
func (v EmbeddingVector) NormedCosineSimilarity(u EmbeddingVector) (cs float32, err error) {
	d, err := v.Dot(u)
	if err != nil {
		return 0, err
	}

	return d, nil
}

func (v *EmbeddingVector) Normalise() {
	mag := v.Mag()
	if mag == 0 {
		return
	}
	for i := range *v {
		(*v)[i] /= mag
	}
}
