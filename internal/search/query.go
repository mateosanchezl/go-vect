package search

import (
	"encoding/binary"
	"math"
	"os"

	"github.com/mateosanchezl/go-vect/internal/embedding"
)

// func getSimilar(query string, model embedding.EmbeddingModel) (results []string, err error) {

// }

func getVectors() (evs []embedding.EmbeddingVector, err error) {
	evs = []embedding.EmbeddingVector{}

	data, err := os.ReadFile("internal/db/data.bin")
	if err != nil {
		return evs, err
	}

	bytes_per_vect := 768 * 8

	n_vects := len(data) / bytes_per_vect
	ofs := 0

	for _ = range n_vects {
		vect := []float64{}
		for i := 0; i < bytes_per_vect; i += 8 {
			fl_bytes := data[ofs+i : ofs+i+8]
			bits := binary.LittleEndian.Uint64(fl_bytes)
			vf := math.Float64frombits(bits)
			if err != nil {
				return evs, err
			}
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
