package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"

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

	// model := embedding.HfEmbeddingModel{
	// 	Token:    os.Getenv("HUGGING_FACE_INFERENCE_API_TOKEN"),
	// 	ModelUrl: "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5/pipeline/feature-extraction",
	// }

	model := embedding.MiniLM{}

	for {
		fmt.Print("\nInput text to embed and store, q to quit, s to search: ")
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

		if str == "s" {
			fmt.Print("Input query text: ")
			qs, err := rd.ReadString('\n')
			if err != nil {
				log.Fatal("failed to read query")
			}

			qs = strings.TrimSpace(string(qs))
			_, err = search.GetSimilar(qs, &model)
			if err != nil {
				log.Fatal(err)
			}

		} else {
			emb, err := model.Embed(str)
			if err != nil {
				log.Fatal("failed to embed:", err)
			}

			storage.StoreEmbedding(emb)
			err = storage.StoreEmbeddingMetaData(storage.EmbeddingMetaData{
				Offset: len(emb),
				Text:   str,
			})
			if err != nil {
				log.Fatal("failed to store embedding metadata:", err)
			}

			fmt.Println("✓ Embedding stored successfully")

			// 	for _, em := range embeddings {
			// 		storage.StoreEmbedding(em)
			// 		fmt.Println("Successfully stored embedding")
			// 	}

			// 	for i, ch := range chunks {
			// 		ofs, err := storage.GetLastOffset()
			// 		if err != nil {
			// 			log.Fatal("failed to get last offset:", err)
			// 		}

			// 		ofs += (len(embeddings[i]) * 8)
			// 		md := storage.EmbeddingMetaData{
			// 			Offset: ofs,
			// 			Text:   ch,
			// 		}

			// 		err = storage.StoreEmbeddingMetaData(md)
			// 		if err != nil {
			// 			log.Fatal("failed to store embedding metadata:", err)
			// 		}

			// 		fmt.Printf("Using offset: %v\n", ofs)
			// 	}
			// }
		}
	}
}
