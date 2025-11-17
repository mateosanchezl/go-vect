package storage

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"strings"

	"github.com/mateosanchezl/go-vect/internal/embedding"
)

// Appends embedding to data file
func StoreEmbedding(embedding embedding.EmbeddingVector, text string) {
	bs := vectorToByteSlice(embedding)

	file, err := os.OpenFile("internal/db/data.bin", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatal("failed to open data file:", err)
	}
	defer file.Close()

	_, err = file.Write(bs)
	if err != nil {
		log.Fatal("failed to write embedding to data file:", err)
	}

	// Store embedding metadata
	err = storeEmbeddingMetaData(embedding, text)
	if err != nil {
		log.Fatalln("error storing embedding metadata: ", err)
	}
}

// Turns an embedding vector into a single byte slice
func vectorToByteSlice(v embedding.EmbeddingVector) []byte {
	out := make([]byte, len(v)*4) // Allocate 4 bytes to each float in vector

	for i := range len(v) {
		bits := math.Float32bits(v[i])                        // Turn the float to bits
		binary.LittleEndian.PutUint32(out[i*4:(i+1)*4], bits) // Put it in the byte slice
	}

	return out
}

// Store metadata
type EmbeddingMetaData struct {
	Offset int
	Text   string
}

func storeEmbeddingMetaData(embedding embedding.EmbeddingVector, text string) (err error) {
	file, err := os.OpenFile("internal/db/metadata.jsonl", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()

	ofs := calculateOffset(embedding)

	md := EmbeddingMetaData{
		Offset: ofs,
		Text:   text,
	}

	mdJson, err := json.Marshal(md)
	if err != nil {
		return err
	}

	_, err = file.WriteString(string(mdJson) + "\n")
	if err != nil {
		return err
	}

	fmt.Println("wrote: ", string(mdJson))

	return nil
}

func calculateOffset(embedding embedding.EmbeddingVector) int {
	last, err := getLastOffset()
	if err != nil {
		log.Fatalln(err)
	}

	byteCount := len(embedding) * 4
	return last + byteCount
}

// Gets the last offset recorded in metadata if available
func getLastOffset() (offset int, err error) {
	data, err := os.ReadFile("internal/db/metadata.jsonl")
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil // File doesn't exist, start at 0
		}
		return 0, err // Real error
	}

	if len(data) == 0 {
		return 0, nil
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) == 0 {
		return 0, nil
	}

	last := lines[len(lines)-1]

	var lastMd EmbeddingMetaData

	err = json.Unmarshal([]byte(last), &lastMd)
	if err != nil {
		return 0, err
	}

	return lastMd.Offset, nil
}

func ClearData() {
	paths := []string{"internal/db/metadata.jsonl", "internal/db/data.bin"}
	for _, p := range paths {
		f, err := os.OpenFile(p, os.O_RDWR|os.O_TRUNC, 0644)
		if err != nil {
			log.Fatalln("error clearing data files: ", err)
		}
		defer f.Close()
	}
}
