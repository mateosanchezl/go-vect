package embedding

import (
	"fmt"
	"log"
	"time"

	"github.com/mateosanchezl/go-vect/internal/tokenizer"
	ort "github.com/yalue/onnxruntime_go"
)

type MiniLM struct{}

func (m *MiniLM) Embed(chunk string) (embedding EmbeddingVector, err error) {
	enc := tokenizer.Encode(chunk, true)
	tokens := enc.Tokens
	attentionMask := enc.AttentionMask
	tIds := enc.TypeIds

	start := time.Now()
	// Create input id tensor
	inputIds, err := ort.NewTensor(ort.NewShape(1, tokens.Length), tokens.Ids)
	if err != nil {
		log.Fatal("error creating input id tensor: ", err)
	}
	defer inputIds.Destroy()

	// Create attention mask tensor
	attMask, err := ort.NewTensor(ort.NewShape(1, attentionMask.Length), attentionMask.Mask)
	if err != nil {
		log.Fatal("error creating attention mask tensor: ", err)
	}
	defer attMask.Destroy()

	// Create token type id tensor
	typeIds, err := ort.NewTensor(ort.NewShape(1, tIds.Length), tIds.Ids)
	if err != nil {
		log.Fatal("error creating type id tensor: ", err)
	}
	defer typeIds.Destroy()

	// Create output tensor
	outputShape := ort.NewShape(1, tokens.Length, 384) // batch size, sequence length, hidden size (from mini lm)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Fatal("error creating output tensor: ", err)
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession("models/all-MiniLM-L6-v2/onnx/model.onnx",
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		[]ort.Value{inputIds, attMask, typeIds},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		log.Fatal("failed creating session with model.onnx: ", err)
	}
	defer session.Destroy()

	err = session.Run()
	if err != nil {
		log.Fatal("failed to run session: ", err)
	}

	elapsed := time.Since(start)
	fmt.Println("Embedded in", elapsed.Milliseconds(), "ms")

	outputData := outputTensor.GetData()
	pooled := meanPoolSingle(outputData, 384, int(tokens.Length), attentionMask.Mask)

	validated, err := NewVector(pooled, 384)
	if err != nil {
		return EmbeddingVector{}, err
	}

	return validated, nil
}

func (m *MiniLM) EmbedBatch(chunks []string) (embeddings []EmbeddingVector, err error) {
	// Tokenize and prepare chunks
	encs := tokenizer.EncodeBatch(chunks, true)

	// Create input tensors
	inputIds, err := ort.NewTensor(ort.NewShape(encs.BatchSize, encs.SequenceLength), encs.TokenIds)
	if err != nil {
		log.Fatal("error creating input id tensor: ", err)
	}
	defer inputIds.Destroy()

	attMask, err := ort.NewTensor(ort.NewShape(encs.BatchSize, encs.SequenceLength), encs.AttentionMask)
	if err != nil {
		log.Fatal("error creating attention mask tensor: ", err)
	}
	defer attMask.Destroy()

	typeIds, err := ort.NewTensor(ort.NewShape(encs.BatchSize, encs.SequenceLength), encs.TypeIds)
	if err != nil {
		log.Fatal("error creating type id tensor: ", err)
	}
	defer typeIds.Destroy()

	// Create output tensor
	outputShape := ort.NewShape(encs.BatchSize, encs.SequenceLength, 384) // batch size, sequence length, hidden size (from mini lm)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Fatal("error creating output tensor: ", err)
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession("models/all-MiniLM-L6-v2/onnx/model.onnx",
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		[]ort.Value{inputIds, attMask, typeIds},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		log.Fatal("failed creating session with model.onnx: ", err)
	}
	defer session.Destroy()

	err = session.Run()
	if err != nil {
		log.Fatal("failed to run session: ", err)
	}

	return embeddings, nil
}

// Apply attention masked mean pooling for a single embedding output, assumed batch size 1
func meanPoolSingle(rawOut []float32, hiddenSize int, seqLength int, attentionMask []int64) []float32 {
	start := time.Now()
	// Split output into vector embedding per token
	split := make([][]float32, seqLength)
	for i := range seqLength {
		split[i] = rawOut[i*hiddenSize : (i+1)*hiddenSize]
	}

	// Get valid token indices
	r := make([]int, 0)
	for i := range attentionMask {
		if attentionMask[i] == 1 {
			r = append(r, i)
		}
	}

	out := make([]float32, hiddenSize)

	for i := range hiddenSize {
		var sum float32
		for _, idx := range r {
			sum += split[idx][i]
		}
		out[i] = sum / float32(len(r))
	}
	elapsed := time.Since(start)
	fmt.Println("Mean pooled in", elapsed.Milliseconds(), "ms")
	return out
}
