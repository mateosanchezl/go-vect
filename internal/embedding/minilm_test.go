package embedding

import "testing"

func TestMeanPoolSingle(t *testing.T) {
	hiddenSize := 3
	seqLength := 4
	rawOut := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}
	attentionMask := []int64{1, 1, 0, 0}

	pool := meanPoolSingle(rawOut, hiddenSize, seqLength, attentionMask)

	expected := []float32{2.5, 3.5, 4.5}
	for i := range expected {
		if pool[i] != expected[i] {
			t.Fatalf("index %d: expected %v, got %v", i, expected[i], pool[i])
		}
	}
}

func TestMeanPoolBatch(t *testing.T) {
	batchSize := 2
	hiddenSize := 2
	seqLength := 3
	rawOut := []float32{
		1, 1,
		2, 2,
		3, 3,
		4, 4,
		5, 5,
		6, 6,
	}
	attentionMasks := [][]int64{
		{1, 0, 1},
		{0, 1, 1},
	}

	pool := meanPoolBatch(rawOut, batchSize, hiddenSize, seqLength, attentionMasks)

	expected := [][]float32{
		{2, 2},
		{5.5, 5.5},
	}
	for i := range expected {
		for j := range expected[i] {
			if pool[i][j] != expected[i][j] {
				t.Fatalf("vector %d idx %d: expected %v, got %v", i, j, expected[i][j], pool[i][j])
			}
		}
	}
}
