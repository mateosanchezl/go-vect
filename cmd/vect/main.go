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
		log.Fatal("failed to load config:", err)
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
		log.Fatal("failed to embed batch:", err)
	}
	fmt.Printf("Embeddings length: %v\n", len(embeddings))

	for _, em := range embeddings {
		storage.StoreEmbedding(em)
		fmt.Println("Successfully stored embedding")
	}

	for i, ch := range chunks {
		ofs, err := storage.GetLastOffset()
		if err != nil {
			log.Fatal("failed to get last offset:", err)
		}

		ofs += (len(embeddings[i]) * 8)
		md := storage.EmbeddingMetaData{
			Offset: ofs,
			Text:   ch,
		}

		err = storage.StoreEmbeddingMetaData(md)
		if err != nil {
			log.Fatal("failed to store embedding metadata:", err)
		}

		fmt.Printf("Using offset: %v\n", ofs)
	}
}
