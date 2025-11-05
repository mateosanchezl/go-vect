package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/joho/godotenv"
)

type EmbeddingRequest struct {
	Inputs string `json:"inputs"`
}

func GetEmbedding(text string){
	err := godotenv.Load()
	if err != nil {
		log.Fatal(err)
	}
	hfToken := os.Getenv("HUGGING_FACE_INFERENCE_API_TOKEN")

	// Create JSON payload
	payload := EmbeddingRequest {
		Inputs: text,
	}
	jsonData, err := json.Marshal(payload)

	if err != nil {
		log.Fatal(err)
	}

	url := "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5/pipeline/feature-extraction"

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	req.Header.Set("Authorization", "Bearer " + hfToken)
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

	fmt.Printf("Response body: %v", string(body))

	if err != nil {
		log.Fatal(err)
	}
}