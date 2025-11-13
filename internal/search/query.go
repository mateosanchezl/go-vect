package search

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/mateosanchezl/go-vect/internal/embedding"
)

type similarityResult struct {
	similarity float64
	pos        int
}

func GetSimilar(query string, model embedding.EmbeddingModel) (results []string, err error) {
	evs, err := getVectors()
	if err != nil {
		return nil, err
	}

	qv, err := model.Embed(query)
	if err != nil {
		return nil, err
	}
	scores := make([]similarityResult, 0)

	worst := similarityResult{
		1.0, 1,
	}
	best := similarityResult{
		0.0, 0,
	}

	for i, cv := range evs {
		sim, err := CosineSimilarity(cv, qv)
		if err != nil {
			return nil, err
		}
		rs := similarityResult{
			similarity: sim,
			pos:        i,
		}
		if sim > best.similarity {
			best.similarity = sim
			best.pos = i
		}
		if sim < worst.similarity {
			worst.similarity = sim
			worst.pos = i
		}
		scores = append(scores, rs)
	}

	fmt.Printf("Highest score found: %v\n at vector position %v\n", best.similarity, best.pos)
	fmt.Printf("Lowest score found: %v\n at vector position %v\n", worst.similarity, worst.pos)
	return
}

func getVectors() (evs []embedding.EmbeddingVector, err error) {
	evs = []embedding.EmbeddingVector{}

	data, err := os.ReadFile("internal/db/data.bin")
	if err != nil {
		return evs, err
	}

	bytes_per_vect := 768 * 8

	n_vects := len(data) / bytes_per_vect
	ofs := 0

	for range n_vects {
		vect := []float64{}
		for i := 0; i < bytes_per_vect; i += 8 {
			fl_bytes := data[ofs+i : ofs+i+8]
			bits := binary.LittleEndian.Uint64(fl_bytes)
			vf := math.Float64frombits(bits)
			vect = append(vect, vf)
		}
		v, err := embedding.NewVector(vect)
		if err != nil {
			return evs, err
		}

		evs = append(evs, v)
		ofs += bytes_per_vect
	}

	return evs, nil
}
