package main

import (
	"flag"

	"github.com/mateosanchezl/go-vect/services"
)

func main(){
	text := flag.String("text", "", "Text to embed")

	flag.Parse()

	services.GetEmbedding(*text)
}

