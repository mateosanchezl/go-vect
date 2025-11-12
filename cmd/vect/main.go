package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

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

	for {
		fmt.Println("Input text to embed and store, q to quit: ")
		rd := bufio.NewReader(os.Stdin)

		text, err := rd.ReadString('\n')
		if err != nil {
			log.Fatal("failed to read file (main):", err)
		}

		str := strings.TrimSpace(string(text))
		if str == "q" {
			fmt.Println("Bye!")
			os.Exit(0)
		}

		chunker := chunking.FixedChunker{
			ChunkSize: 512,
		}
		chunks := chunker.Chunk(str)

		model := embedding.HfEmbeddingModel{
			Token:    os.Getenv("HUGGING_FACE_INFERENCE_API_TOKEN"),
			ModelUrl: "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5/pipeline/feature-extraction",
		}

		start := time.Now()
		embeddings, err := model.EmbedBatch(chunks)
		if err != nil {
			log.Fatal("failed to embed batch:", err)
		}
		elapsed := time.Since(start)
		fmt.Printf("Embed time: %v ms\n", elapsed.Milliseconds())

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
}
