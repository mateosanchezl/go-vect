package tokenizer

import (
	"log"

	"github.com/mateosanchezl/go-vect/internal/config"
	"github.com/sugarme/tokenizer"
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

type Encoding struct {
	Tokens        Tokens
	AttentionMask AttentionMask
	TypeIds       TypeIds
}

type EncodedBatch struct {
	TokenIds       []int64
	AttentionMask  []int64
	TypeIds        []int64
	BatchSize      int64
	SequenceLength int64
}

func getTokenizer() *tokenizer.Tokenizer {
	if config.Tokenizer == nil {
		panic("tokenizer not initialised. ensure config is loaded. see config.LoadConfig()")
	}
	return config.Tokenizer
}

func EncodeBatch(texts []string, withSpecialTokens bool) (encodedBatch EncodedBatch) {
	tk := getTokenizer()

	inputs := make([]tokenizer.EncodeInput, len(texts))

	for i, text := range texts {
		seq := tokenizer.NewInputSequence(text)
		inputs[i] = tokenizer.NewSingleEncodeInput(seq)
	}

	encs, err := tk.EncodeBatch(inputs, withSpecialTokens)
	if err != nil {
		log.Fatal(err)
	}

	n := len(encs)
	if n == 0 {
		log.Fatal("no encodings returned from tokenizer.EncodeBatch")
	}

	sl := len(encs[0].GetIds())

	// Flatten encodings into single slice
	tokenIds := make([]int64, n*sl)
	attentionMask := make([]int64, n*sl)
	typeIds := make([]int64, n*sl)

	for i, enc := range encs {
		startIdx := i * sl
		for j := range sl {
			tokenIds[startIdx+j] = int64(enc.Ids[j])
			attentionMask[startIdx+j] = int64(enc.AttentionMask[j])
			typeIds[startIdx+j] = int64(enc.TypeIds[j])
		}
	}

	return EncodedBatch{
		TokenIds:       tokenIds,
		AttentionMask:  attentionMask,
		TypeIds:        typeIds,
		BatchSize:      int64(n),
		SequenceLength: int64(sl),
	}
}

func Encode(text string, withSpecialTokens bool) (encoding Encoding) {
	tk := getTokenizer()
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

	return Encoding{
		Tokens:        tks,
		AttentionMask: am,
		TypeIds:       tIds,
	}
}

func intArrtoInt64Arr(arr []int) (conv []int64) {
	out := make([]int64, len(arr))
	for i, v := range arr {
		out[i] = int64(v)
	}
	return out
}
