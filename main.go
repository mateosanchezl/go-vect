package main

import (
	"flag"
	"fmt"

	"github.com/mateosanchezl/go-vect/services"
)

func main() {
	text := flag.String("text", "", "Text to embed")

	flag.Parse()

	embedding := services.GetEmbedding(*text)

	services.StoreEmbedding(embedding)

	fmt.Println(embedding)
}
