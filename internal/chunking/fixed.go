package chunking

type FixedChunker struct {
	ChunkSize int // number of chars
}

func (fc *FixedChunker) Chunk(text string) (chunks []string) {
	if fc.ChunkSize <= 0 {
		return []string{text}
	}

	runes := []rune(text)
	if len(runes) == 0 {
		return []string{}
	}

	if len(runes) <= fc.ChunkSize {
		return []string{text}
	}

	out := make([]string, 0, (len(runes)+fc.ChunkSize-1)/fc.ChunkSize)
	for i := 0; i < len(runes); i += fc.ChunkSize {
		end := min(len(runes), i+fc.ChunkSize)
		out = append(out, string(runes[i:end]))
	}

	return out
}
