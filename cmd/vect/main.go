package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/mateosanchezl/go-vect/internal/chunking"
	"github.com/mateosanchezl/go-vect/internal/config"
	"github.com/mateosanchezl/go-vect/internal/embedding"
)

func main() {
	err := config.LoadConfig()
	if err != nil {
		log.Fatal(err)
	}

	ch := chunking.FixedChunker{
		ChunkSize: 512,
	}

	chunks := ch.Chunk(`The future of artificial intelligence lies not only in larger models, but in smaller, specialized ones that understand context deeply. As organizations begin integrating AI into daily workflows, efficiency becomes more important than scale.

For instance, a law firm might deploy a fine-tuned model that summarizes case histories and extracts relevant precedents, while a logistics company could use a lightweight LLM to forecast shipping delays. These domain-focused systems often outperform general-purpose models when the task boundaries are clear.

Yet, specialization introduces a trade-off: fragmentation. Without shared embeddings or a unifying vector database, each model becomes an island of knowledge. This is where semantic search, embeddings, and retrieval-augmented generation (RAG) architectures bridge the gap.

A vector database doesn’t just store information — it maps meaning. Text, images, or even audio snippets can be projected into a common semantic space where “closeness” implies conceptual similarity. When a user queries the system, the database retrieves the most relevant chunks based on this latent meaning, not simple keywords.

The challenge then becomes chunking: how to split long documents into meaningful segments without breaking semantic coherence. Too short, and context is lost. Too long, and embeddings become diluted. A balanced chunk preserves narrative flow while staying within token limits, ensuring embeddings reflect both detail and intent.`)

	text := flag.String("text", "", "Text to embed")
	flag.Parse()

	chunks = []string{*text}

	model := embedding.HfEmbeddingModel{
		Token:    os.Getenv("HUGGING_FACE_INFERENCE_API_TOKEN"),
		ModelUrl: "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5/pipeline/feature-extraction",
	}

	e, err := model.Embed(chunks)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(e)
}
