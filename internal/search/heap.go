package search

/*
A min heap implementation for efficient top k search in query.go.
Note that after sorting, a fresh heap needs to be created for next search.
*/
type MinHeap struct {
	K int
	H []SimilarityResult
	N int
}

func (mh *MinHeap) Init(K int) {
	mh.K = K
	mh.N = 0
	mh.H = make([]SimilarityResult, 0, K)
}

func (mh *MinHeap) bubbleUp(i int) {
	for i > 0 {
		p := mh.H[(i-1)/2]
		if p.CosSim > mh.H[i].CosSim {
			mh.H[i], mh.H[(i-1)/2] = mh.H[(i-1)/2], mh.H[i]
		} else {
			break
		}
		i = (i - 1) / 2
	}
}

func (mh *MinHeap) bubbleDown(i int) {
	li := 2*i + 1
	ri := 2*i + 2
	smallest := i

	if li < mh.N && mh.H[li].CosSim < mh.H[smallest].CosSim {
		smallest = li
	}
	if ri < mh.N && mh.H[ri].CosSim < mh.H[smallest].CosSim {
		smallest = ri
	}

	if smallest != i {
		mh.H[i], mh.H[smallest] = mh.H[smallest], mh.H[i]
		mh.bubbleDown(smallest)
	}
}

func (mh *MinHeap) Insert(x SimilarityResult) {
	if mh.K == 0 {
		return
	}

	if len(mh.H) < mh.K {
		mh.H = append(mh.H, x)
		mh.N++
		mh.bubbleUp(len(mh.H) - 1)
	} else {
		if x.CosSim > mh.H[0].CosSim {
			mh.H[0] = x
			mh.bubbleDown(0)
		}
	}
}

func (mh *MinHeap) Sort() {
	r := 0
	for mh.N > 1 {
		l := mh.N - 1
		mh.N--
		// Swap root with last
		mh.H[r], mh.H[l] = mh.H[l], mh.H[r]
		// Last is sorted so size decreases
		mh.bubbleDown(r)
	}
}
