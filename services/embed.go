package services

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/joho/godotenv"
	"github.com/mateosanchezl/go-vect/types"
)

type EmbeddingRequest struct {
	Inputs string `json:"inputs"`
}

func GetEmbedding(text string) types.EmbeddingVector {
	err := godotenv.Load()
	if err != nil {
		log.Fatal(err)
	}
	hfToken := os.Getenv("HUGGING_FACE_INFERENCE_API_TOKEN")

	payload := EmbeddingRequest{
		Inputs: text,
	}
	jsonData, err := json.Marshal(payload)

	if err != nil {
		log.Fatal(err)
	}

	url := "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5/pipeline/feature-extraction"

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	req.Header.Set("Authorization", "Bearer "+hfToken)
	req.Header.Set("Content-Type", "application/json")

	if err != nil {
		log.Fatal(err)
	}

	client := http.DefaultClient

	res, err := client.Do(req)

	if err != nil {
		log.Fatal(err)
	}

	body, err := io.ReadAll(res.Body)

	if err != nil {
		log.Fatal(err)
	}

	var vect []float64

	json.Unmarshal(body, &vect)

	validated, err := types.NewVector(vect)

	if err != nil {
		log.Fatal(err)
	}

	return validated
}
