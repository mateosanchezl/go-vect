package vect

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/mateosanchezl/go-vect/internal/config"
	"github.com/mateosanchezl/go-vect/internal/embedding"
)

func main() {
	err := config.LoadConfig()
	if(err != nil){
		log.Fatal(err)
	}

	text := flag.String("text", "", "Text to embed")
	flag.Parse()

	chunks := []string{*text}

	model := embedding.HfEmbeddingModel{
		Token:    os.Getenv("HUGGING_FACE_INFERENCE_API_TOKEN"),
		ModelUrl: "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5/pipeline/feature-extraction",
	}

	// services.StoreEmbedding(embeddings)

	fmt.Println(embeddings[0])
}
