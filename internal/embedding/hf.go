package embedding

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
)

type HfEmbeddingRequest struct {
	Inputs []string `json:"inputs"`
}

type HfEmbeddingModel struct {
	Token    string
	ModelUrl string
}

func (hf *HfEmbeddingModel) sendEmbeddingRequest(payload HfEmbeddingRequest) (body []byte, err error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", hf.ModelUrl, bytes.NewBuffer(jsonData))
	req.Header.Set("Authorization", "Bearer "+hf.Token)
	req.Header.Set("Content-Type", "application/json")
	if err != nil {
		return nil, err
	}

	client := http.DefaultClient

	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	body, err = io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}

	return body, nil
}

func (hf *HfEmbeddingModel) Embed(chunk []string) (embedding EmbeddingVector, err error) {
	payload := HfEmbeddingRequest{
		Inputs: chunk,
	}

	body, err := hf.sendEmbeddingRequest(payload)
	if err != nil {
		return nil, err
	}

	var vect []float64
	json.Unmarshal(body, &vect)

	validated, err := NewVector(vect)
	if err != nil {
		return nil, err
	}

	return validated, nil
}

func (hf *HfEmbeddingModel) EmbedBatch(chunks []string) (embeddings []EmbeddingVector, err error) {
	payload := HfEmbeddingRequest{
		Inputs: chunks,
	}
	body, err := hf.sendEmbeddingRequest(payload)

	var vects [][]float64
	json.Unmarshal(body, &vects)

	validatedVects := []EmbeddingVector{}

	for _, v := range vects {
		validated, err := NewVector(v)
		if err != nil {
			return nil, err
		}
		validatedVects = append(validatedVects, validated)
	}

	if len(validatedVects) == 0 {
		return nil, errors.New("no embedding vectors were parsed")
	}

	return validatedVects, nil
}
