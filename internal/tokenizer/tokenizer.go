package tokenizer

import (
	"log"
	"os"

	"github.com/sugarme/tokenizer/pretrained"
)

func Encode(text string) (ids []int, tokens []string) {
	tokenizerPath := os.Getenv("TOKENIZER_JSON")
	if tokenizerPath == "" {
		log.Fatal("failed getting TOKENIZER_JSON env variable, did you run make setup?")
	}

	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		log.Fatal("error tokenising from pretrained: ", err)
	}

	en, err := tk.EncodeSingle(text)
	if err != nil {
		log.Fatal(err)
	}

	return en.Ids, en.Tokens
}
