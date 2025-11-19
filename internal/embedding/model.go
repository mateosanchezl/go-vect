package embedding

// Model interface
type EmbeddingModel interface {
	Embed(chunk string) (embedding EmbeddingVector, err error)
	EmbedBatch(chunks []string) (embeddings []EmbeddingVector, err error)
}
