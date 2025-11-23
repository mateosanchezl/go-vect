package search

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"strings"

	"github.com/mateosanchezl/go-vect/internal/embedding"
	"github.com/mateosanchezl/go-vect/internal/storage"
)

type SimilarityResult struct {
	CosSim float32
	Pos    int
}

type TopKSearchResult struct {
	CosSim float32
	Text   string
}

func SearchTopKSimilar(query string, k int, model embedding.EmbeddingModel) (results []TopKSearchResult, err error) {
	evs, err := readVectors()
	if err != nil {
		return nil, err
	}

	qv, err := model.Embed(query)
	if err != nil {
		return nil, err
	}
	qv.Normalise()

	mh := MinHeap{}
	mh.Init(k)
	for i, cv := range evs {
		sim, err := cv.NormedCosineSimilarity(qv)
		mh.Insert(SimilarityResult{CosSim: sim, Pos: i})
		if err != nil {
			return nil, err
		}
	}
	mh.Sort()

	md, err := readMetadata()
	if err != nil {
		return nil, err
	}

	out := make([]TopKSearchResult, k)
	for i, rs := range mh.H {
		var record storage.EmbeddingMetaData
		json.Unmarshal([]byte(md[rs.Pos]), &record)
		out[i] = TopKSearchResult{
			Text:   record.Text,
			CosSim: rs.CosSim,
		}
	}

	return out, nil
}

func readVectors() (evs []embedding.EmbeddingVector, err error) {
	evs = []embedding.EmbeddingVector{}

	data, err := os.ReadFile(os.Getenv("VECTOR_DB_PATH"))
	if err != nil {
		return evs, err
	}

	bytes_per_vect := 384 * 4

	n_vects := len(data) / bytes_per_vect
	ofs := 0

	for range n_vects {
		vect := []float32{}
		for i := 0; i < bytes_per_vect; i += 4 {
			fl_bytes := data[ofs+i : ofs+i+4]
			bits := binary.LittleEndian.Uint32(fl_bytes)
			vf := math.Float32frombits(bits)
			vect = append(vect, vf)
		}
		v, err := embedding.NewVector(vect, 384)
		if err != nil {
			return evs, err
		}

		evs = append(evs, v)
		ofs += bytes_per_vect
	}

	return evs, nil
}

func readMetadata() (lines []string, err error) {
	md, err := os.ReadFile(os.Getenv("METADATA_DB_PATH"))
	if err != nil {
		if os.IsNotExist(err) {
			_, err = os.Create(os.Getenv("METADATA_DB_PATH")) // Create file if it doesn't exist
			if err != nil {
				return nil, err
			}
			return []string{}, nil
		}
	}

	if len(md) == 0 {
		return []string{}, nil
	}

	lines = strings.Split(strings.TrimSpace(string(md)), "\n")
	if len(lines) == 0 {
		return []string{}, nil
	}

	return lines, nil
}
