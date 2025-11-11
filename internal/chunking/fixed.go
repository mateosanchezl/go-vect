package chunking

type FixedChunker struct {
	ChunkSize int // number of chars
}

func (fc *FixedChunker) Chunk(text string) (chunks []string) {
	if len(text) < fc.ChunkSize {
		return []string{text}
	}

	out := []string{}
	for i := 0; i < len(text); i += fc.ChunkSize {
		if i+fc.ChunkSize >= len(text) {
			out = append(out, text[i:])
			break
		}
		out = append(out, text[i:i+fc.ChunkSize])
	}

	return out
}
