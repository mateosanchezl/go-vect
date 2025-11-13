package search

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/mateosanchezl/go-vect/internal/embedding"
	"github.com/mateosanchezl/go-vect/internal/storage"
)

type SimilaritySearchResult struct {
	text       string
	similarity float64
}

type similarityResult struct {
	similarity float64
	pos        int
	vect       embedding.EmbeddingVector
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
		1.0, 1, embedding.EmbeddingVector{},
	}
	best := similarityResult{
		0.0, 0, embedding.EmbeddingVector{},
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
			best.vect = cv
		}
		if sim < worst.similarity {
			worst.similarity = sim
			worst.pos = i
			worst.vect = cv
		}
		scores = append(scores, rs)
	}

	md, err := os.ReadFile("internal/db/metadata.jsonl")
	if err != nil {
		if os.IsNotExist(err) {
			_, err = os.Create("internal/db/metadata.jsonl")
			if err != nil {
				return nil, err
			}
			fmt.Println("could not find metadata file, created successfuly.")
			return []string{}, nil
		}
	}

	if len(md) == 0 {
		return []string{}, nil
	}

	lines := strings.Split(strings.TrimSpace(string(md)), "\n")
	if len(lines) == 0 {
		return []string{}, nil
	}

	var bestMd, worstMd storage.EmbeddingMetaData

	json.Unmarshal([]byte(lines[best.pos]), &bestMd)
	json.Unmarshal([]byte(lines[worst.pos]), &worstMd)

	bestRes := SimilaritySearchResult{
		similarity: best.similarity,
		text:       bestMd.Text,
	}

	worstRes := SimilaritySearchResult{
		similarity: worst.similarity,
		text:       worstMd.Text,
	}

	fmt.Printf("\nBest result found: \n%v \nWorst result found: %v", bestRes, worstRes)
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
