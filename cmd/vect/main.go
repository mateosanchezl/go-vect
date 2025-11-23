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
	"github.com/mateosanchezl/go-vect/internal/search"
	"github.com/mateosanchezl/go-vect/internal/storage"
)

func main() {
	err := config.LoadConfig()
	if err != nil {
		log.Fatal("failed to load config:", err)
	}

	chunker := chunking.FixedChunker{
		ChunkSize: 150,
	}

	model := embedding.MiniLM{}

	for {
		fmt.Print("\nInput text to embed and store, f for file embedding, s to search, d to delete data, q to quit : ")
		rd := bufio.NewReader(os.Stdin)

		text, err := rd.ReadString('\n')
		if err != nil {
			log.Fatal("failed to read file (main):", err)
		}

		str := strings.TrimSpace(string(text))
		if str == "f" {
			fmt.Print("Input file path: ")
			filePath, err := rd.ReadString('\n')
			if err != nil {
				log.Fatal("failed to read file path:", err)
			}

			filePath = strings.TrimSpace(filePath)
			f, err := os.ReadFile(filePath)
			if err != nil {
				log.Fatal("failed to read file:", err)
			}

			chunks := chunker.Chunk(string(f))
			start := time.Now()

			embeddings, err := model.EmbedBatch(chunks)
			if err != nil {
				log.Fatal("failed to embed batch:", err)
			}

			elapsed := time.Since(start)
			fmt.Printf("Embedded %d chunks in %s\n", len(chunks), elapsed)

			start = time.Now()
			for i, e := range embeddings {
				storage.StoreEmbedding(e, chunks[i])
			}
			storeElapsed := time.Since(start)
			fmt.Printf("Stored in: %s\n", storeElapsed)
			continue
		}

		if str == "q" {
			fmt.Println("Bye!")
			os.Exit(0)
		}

		if str == "d" {
			err := storage.ClearData()
			if err != nil {
				log.Fatal("failed to clear data:", err)
			}
			fmt.Println("successfully deleted data")
			continue
		}

		if str == "s" {
			fmt.Print("Input query text: ")
			qs, err := rd.ReadString('\n')
			if err != nil {
				log.Fatal("failed to read query:", err)
			}

			qs = strings.TrimSpace(string(qs))
			rs, err := search.SearchTopKSimilar(qs, 10, &model)
			if err != nil {
				log.Fatal("failed to get similar:", err)
			}

			fmt.Println("Results: ", rs)

		} else {
			chunks := chunker.Chunk(str)

			start := time.Now()
			embeddings, err := model.EmbedBatch(chunks)
			for _, embedding := range embeddings {
				fmt.Printf("Embedding: %v\n", len(embedding))
			}
			if err != nil {
				log.Fatal("failed to embed batch:", err)
			}
			elapsed := time.Since(start)
			fmt.Printf("Embedded %d chunks in %s\n", len(chunks), elapsed)

			start = time.Now()
			for i, e := range embeddings {
				storage.StoreEmbedding(e, chunks[i])
			}
			storeElapsed := time.Since(start)
			fmt.Printf("Stored in: %s\n", storeElapsed)
			continue
		}
	}
}
