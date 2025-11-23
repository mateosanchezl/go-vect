package chunking

import (
	"log"
	"strings"
)

type DelimiterChunker struct {
	Delimiter string
}

func (dc *DelimiterChunker) Chunk(text string) (chunks []string) {
	if dc.Delimiter == "" {
		return []string{text}
	}

	if len(dc.Delimiter) != 1 {
		log.Fatal("delimiter must be a single character")
	}

	runes := []rune(text)
	if len(runes) == 0 {
		return []string{}
	}

	return strings.Split(text, dc.Delimiter)
}
