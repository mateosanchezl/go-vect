package tokenizer

import (
	"log"
	"os"

	"github.com/sugarme/tokenizer/pretrained"
)

type Tokens struct {
	Ids    []int64
	Tokens []string
	Length int64
}

type AttentionMask struct {
	Mask   []int64
	Length int64
}

type TypeIds struct {
	Ids    []int64
	Length int64
}

func Encode(text string, withSpecialTokens bool) (tokens Tokens, attentionMask AttentionMask, typeIds TypeIds) {
	tokenizerPath := os.Getenv("TOKENIZER_JSON")
	if tokenizerPath == "" {
		log.Fatal("failed getting TOKENIZER_JSON env variable, did you run make setup?")
	}

	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		log.Fatal("error tokenising from pretrained: ", err)
	}

	en, err := tk.EncodeSingle(text, withSpecialTokens)
	if err != nil {
		log.Fatal(err)
	}

	tks := Tokens{
		Ids:    intArrtoInt64Arr(en.Ids),
		Tokens: en.Tokens,
		Length: int64(len(en.Ids)),
	}

	am := AttentionMask{
		Mask:   intArrtoInt64Arr(en.AttentionMask),
		Length: int64(len(en.AttentionMask)),
	}

	tIds := TypeIds{
		Ids:    intArrtoInt64Arr(en.TypeIds),
		Length: int64(len(en.TypeIds)),
	}

	return tks, am, tIds
}

func intArrtoInt64Arr(arr []int) (conv []int64) {
	out := make([]int64, len(arr))
	for i, v := range arr {
		out[i] = int64(v)
	}
	return out
}
