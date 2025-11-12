package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/mateosanchezl/go-vect/internal/chunking"
	"github.com/mateosanchezl/go-vect/internal/config"
	"github.com/mateosanchezl/go-vect/internal/embedding"
	"github.com/mateosanchezl/go-vect/internal/storage"
)

func main() {
	err := config.LoadConfig()
	if err != nil {
		log.Fatal(err)
	}

	text := flag.String("text", "", "Text to embed")
	flag.Parse()

	res, err := os.ReadFile(*text)

	str := string(res)

	chunker := chunking.FixedChunker{
		ChunkSize: 512,
	}
	chunks := chunker.Chunk(str)

	model := embedding.HfEmbeddingModel{
		Token:    os.Getenv("HUGGING_FACE_INFERENCE_API_TOKEN"),
		ModelUrl: "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5/pipeline/feature-extraction",
	}
	embeddings, err := model.EmbedBatch(chunks)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Embeddings length: %v", len(embeddings))

	ofs := len(embeddings[0]) * 8
	fmt.Printf("Using offset: %v", ofs)

	for _, ch := range chunks {
		md := storage.EmbeddingMetaData{
			Offset: ofs,
			Text:   ch,
		}
		storage.StoreEmbeddingMetaData(md)
	}
}
