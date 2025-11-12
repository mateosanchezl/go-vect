package search

import (
	"errors"
	"math"

	"github.com/mateosanchezl/go-vect/internal/embedding"
)

func dot(u embedding.EmbeddingVector, v embedding.EmbeddingVector) (uv float64, err error) {
	if len(u) != len(v) {
		return 0.0, errors.New("dot: vectors are not of same size")
	}

	t := 0.0
	for i := range len(u) {
		t += u[i] * v[i]
	}
	return t, nil
}

func mag(v embedding.EmbeddingVector) (m float64) {
	m = 0
	for i := range len(v) {
		m += math.Pow(v[i], 2)
	}
	return math.Sqrt(m)
}

func CosineSimilarity(u embedding.EmbeddingVector, v embedding.EmbeddingVector) (cs float64, err error) {
	d, err := dot(u, v)
	if err != nil {
		return 0.0, err
	}

	mu := mag(u)
	mv := mag(v)
	return d / (mu * mv), nil
}
