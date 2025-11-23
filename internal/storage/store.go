package storage

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/mateosanchezl/go-vect/internal/embedding"
)

// Appends embedding to data file
func StoreEmbedding(embedding embedding.EmbeddingVector, text string) error {
	embedding.Normalise()

	bs := vectorToByteSlice(embedding)

	file, err := os.OpenFile(os.Getenv("VECTOR_DB_PATH"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("failed to open data file: %w", err)
	}
	defer file.Close()

	_, err = file.Write(bs)
	if err != nil {
		return fmt.Errorf("failed to write embedding to data file: %w", err)
	}

	// Store embedding metadata
	err = storeEmbeddingMetaData(embedding, text)
	if err != nil {
		return fmt.Errorf("failed to store embedding metadata: %w", err)
	}

	return nil
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
	file, err := os.OpenFile(os.Getenv("METADATA_DB_PATH"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()

	ofs, err := calculateOffset(embedding)
	if err != nil {
		return fmt.Errorf("failed to calculate offset: %w", err)
	}

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

	return nil
}

func calculateOffset(embedding embedding.EmbeddingVector) (int, error) {
	last, err := getLastOffset()
	if err != nil {
		return 0, fmt.Errorf("failed to get last offset: %w", err)
	}

	byteCount := len(embedding) * 4
	return last + byteCount, nil
}

// Gets the last offset recorded in metadata if available
func getLastOffset() (offset int, err error) {
	data, err := os.ReadFile(os.Getenv("METADATA_DB_PATH"))
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

func ClearData() error {
	paths := []string{os.Getenv("VECTOR_DB_PATH"), os.Getenv("METADATA_DB_PATH")}
	for _, p := range paths {
		f, err := os.OpenFile(p, os.O_RDWR|os.O_TRUNC, 0o644)
		if err != nil {
			return fmt.Errorf("failed to clear data file %s: %w", p, err)
		}
		defer f.Close()
	}
	return nil
}
