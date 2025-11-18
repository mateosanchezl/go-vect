package tokenizer

import (
	"fmt"

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
	TokenIds                [][]int64
	AttentionMasks          [][]int64
	TypeIds                 [][]int64
	FlattenedTokenIds       []int64
	FlattenedAttentionMasks []int64
	FlattenedTypeIds        []int64
	BatchSize               int64
	SequenceLength          int64
}

func getTokenizer() *tokenizer.Tokenizer {
	if config.Tokenizer == nil {
		panic("tokenizer not initialised. ensure config is loaded. see config.LoadConfig()")
	}
	return config.Tokenizer
}

func EncodeBatch(texts []string, withSpecialTokens bool) (encodedBatch EncodedBatch, err error) {
	tk := getTokenizer()

	inputs := make([]tokenizer.EncodeInput, len(texts))

	for i, text := range texts {
		seq := tokenizer.NewInputSequence(text)
		inputs[i] = tokenizer.NewSingleEncodeInput(seq)
	}

	encs, err := tk.EncodeBatch(inputs, withSpecialTokens)
	if err != nil {
		return EncodedBatch{}, fmt.Errorf("failed to encode batch: %w", err)
	}

	n := len(encs)
	if n == 0 {
		return EncodedBatch{}, fmt.Errorf("no encodings returned from tokenizer.EncodeBatch")
	}

	sl := len(encs[0].GetIds())

	tokenIds := make([][]int64, n)
	attentionMasks := make([][]int64, n)
	typeIds := make([][]int64, n)
	// Flatten encodings into single slice
	flattenedTokenIds := make([]int64, n*sl)
	flattenedAttentionMask := make([]int64, n*sl)
	flattenedTypeIds := make([]int64, n*sl)

	for i, enc := range encs {
		startIdx := i * sl
		tokenIds[i] = intArrtoInt64Arr(enc.Ids)
		attentionMasks[i] = intArrtoInt64Arr(enc.AttentionMask)
		typeIds[i] = intArrtoInt64Arr(enc.TypeIds)
		for j := range sl {
			flattenedTokenIds[startIdx+j] = int64(enc.Ids[j])
			flattenedAttentionMask[startIdx+j] = int64(enc.AttentionMask[j])
			flattenedTypeIds[startIdx+j] = int64(enc.TypeIds[j])
		}
	}

	return EncodedBatch{
		FlattenedTokenIds:       flattenedTokenIds,
		FlattenedAttentionMasks: flattenedAttentionMask,
		FlattenedTypeIds:        flattenedTypeIds,
		TokenIds:                tokenIds,
		AttentionMasks:          attentionMasks,
		TypeIds:                 typeIds,
		BatchSize:               int64(n),
		SequenceLength:          int64(sl),
	}, nil
}

func Encode(text string, withSpecialTokens bool) (encoding Encoding, err error) {
	tk := getTokenizer()
	en, err := tk.EncodeSingle(text, withSpecialTokens)
	if err != nil {
		return Encoding{}, fmt.Errorf("failed to encode text: %w", err)
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
	}, nil
}

func intArrtoInt64Arr(arr []int) (conv []int64) {
	out := make([]int64, len(arr))
	for i, v := range arr {
		out[i] = int64(v)
	}
	return out
}
