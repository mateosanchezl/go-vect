package embedding

import (
	"math"
	"testing"
)

func TestNewVectorLengthValidation(t *testing.T) {
	vals := []float32{1, 2, 3}

	if _, err := NewVector(vals, len(vals)); err != nil {
		t.Fatalf("expected vector to be created, got error: %v", err)
	}

	if _, err := NewVector(vals, len(vals)+1); err == nil {
		t.Fatalf("expected error for mismatched length")
	}

	if _, err := NewVector([]float32{}, 0); err == nil {
		t.Fatalf("expected error for empty vector")
	}
}

func TestDot(t *testing.T) {
	v := EmbeddingVector{1, 2, 3}
	u := EmbeddingVector{4, 5, 6}

	got, err := v.Dot(u)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	want := float32(32)
	if got != want {
		t.Fatalf("dot product: want %v, got %v", want, got)
	}
}

func TestDotDifferentLengths(t *testing.T) {
	v := EmbeddingVector{1, 2}
	u := EmbeddingVector{3}

	if _, err := v.Dot(u); err == nil {
		t.Fatalf("expected error for different vector sizes")
	}
}

func TestMagnitude(t *testing.T) {
	v := EmbeddingVector{3, 4}
	want := float32(5)

	got := v.Mag()
	if got != want {
		t.Fatalf("magnitude: want %v, got %v", want, got)
	}
}

func TestNormalise(t *testing.T) {
	v := EmbeddingVector{3, 4}
	v.Normalise()

	if math.Abs(float64(v.Mag()-1)) > 1e-6 {
		t.Fatalf("expected normalised vector to have magnitude 1, got %v", v.Mag())
	}
}

func TestNormedCosineSimilarity(t *testing.T) {
	v := EmbeddingVector{1, 0}
	u := EmbeddingVector{1, 0}

	cs, err := v.NormedCosineSimilarity(u)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cs != 1 {
		t.Fatalf("expected cosine similarity of 1, got %v", cs)
	}

	w := EmbeddingVector{0, 1}
	cs, err = v.NormedCosineSimilarity(w)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cs != 0 {
		t.Fatalf("expected cosine similarity of 0, got %v", cs)
	}
}
