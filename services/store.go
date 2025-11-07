package services

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/mateosanchezl/go-vect/types"
)

// Appends embedding to data file
func StoreEmbedding(embedding types.EmbeddingVector) {
	bs := vectorToByteSlice(embedding)

	file, err := os.OpenFile("storage/data.dta", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	wr, err := file.Write(bs)

	fmt.Printf("%v bytes written", wr)
	if err != nil {
		log.Fatal(err)
	}
}

// Turns an embedding vector into a single byte slice
func vectorToByteSlice(v types.EmbeddingVector) []byte {
	out := make([]byte, len(v)*8) // Allocate 8 bytes to each float in vector

	for i := range len(v) {
		bits := math.Float64bits(v[i])                        // Turn the float to bits
		binary.LittleEndian.PutUint64(out[i*8:(i+1)*8], bits) // Put it in the byte slice
	}

	return out
}
