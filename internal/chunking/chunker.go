package chunking

type Chunker interface {
	Chunk(text string) (chunks []string)
}
