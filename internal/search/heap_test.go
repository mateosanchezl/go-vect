package search

import "testing"

func TestMinHeapMaintainsTopK(t *testing.T) {
	mh := MinHeap{}
	mh.Init(3)

	values := []float32{0.1, 0.8, 0.3, 0.9, 0.5, 0.7}
	for i, v := range values {
		mh.Insert(SimilarityResult{CosSim: v, Pos: i})
	}

	if len(mh.H) != 3 {
		t.Fatalf("expected heap to keep 3 items, got %d", len(mh.H))
	}

	mh.Sort()

	expected := []float32{0.9, 0.8, 0.7}
	for i, exp := range expected {
		if mh.H[i].CosSim != exp {
			t.Fatalf("index %d: expected %v, got %v", i, exp, mh.H[i].CosSim)
		}
	}
}
