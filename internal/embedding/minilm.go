package embedding

import (
	"fmt"
	"os"
	"sync"

	"github.com/mateosanchezl/go-vect/internal/tokenizer"
	ort "github.com/yalue/onnxruntime_go"
)

var (
	singleSessionInitOnce sync.Once
	batchSessionInitOnce  sync.Once

	singleSession    *ort.DynamicAdvancedSession
	batchSession     *ort.DynamicAdvancedSession
	singleSessionErr error
	batchSessionErr  error
	singleSessionMu  sync.Mutex
	batchSessionMu   sync.Mutex
)

var (
	modelInputNames  = []string{"input_ids", "attention_mask", "token_type_ids"}
	modelOutputNames = []string{"last_hidden_state"}
)

type MiniLM struct{}

func (m *MiniLM) Embed(chunk string) (embedding EmbeddingVector, err error) {
	enc, err := tokenizer.Encode(chunk, true)
	if err != nil {
		return EmbeddingVector{}, fmt.Errorf("failed to encode chunk: %w", err)
	}
	tokens := enc.Tokens
	attentionMask := enc.AttentionMask
	tIds := enc.TypeIds

	inputIds, err := ort.NewTensor(ort.NewShape(1, tokens.Length), tokens.Ids)
	if err != nil {
		return EmbeddingVector{}, fmt.Errorf("failed to create input id tensor: %w", err)
	}
	defer inputIds.Destroy()

	attMask, err := ort.NewTensor(ort.NewShape(1, attentionMask.Length), attentionMask.Mask)
	if err != nil {
		return EmbeddingVector{}, fmt.Errorf("failed to create attention mask tensor: %w", err)
	}
	defer attMask.Destroy()

	typeIds, err := ort.NewTensor(ort.NewShape(1, tIds.Length), tIds.Ids)
	if err != nil {
		return EmbeddingVector{}, fmt.Errorf("failed to create type id tensor: %w", err)
	}
	defer typeIds.Destroy()

	outputShape := ort.NewShape(1, tokens.Length, 384)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return EmbeddingVector{}, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	session, err := getSingleSession()
	if err != nil {
		return EmbeddingVector{}, err
	}

	inputs := []ort.Value{inputIds, attMask, typeIds}
	outputs := []ort.Value{outputTensor}

	singleSessionMu.Lock()
	err = session.Run(inputs, outputs)
	singleSessionMu.Unlock()
	if err != nil {
		return EmbeddingVector{}, fmt.Errorf("failed to run ONNX session: %w", err)
	}

	outputData := outputTensor.GetData()
	pooled := meanPoolSingle(outputData, 384, int(tokens.Length), attentionMask.Mask)

	validated, err := NewVector(pooled, 384)
	if err != nil {
		return EmbeddingVector{}, err
	}

	return validated, nil
}

func (m *MiniLM) EmbedBatch(chunks []string) (embeddings []EmbeddingVector, err error) {
	encs, err := tokenizer.EncodeBatch(chunks, true)
	if err != nil {
		return nil, fmt.Errorf("failed to encode batch: %w", err)
	}

	inputIds, err := ort.NewTensor(ort.NewShape(encs.BatchSize, encs.SequenceLength), encs.FlattenedTokenIds)
	if err != nil {
		return nil, fmt.Errorf("failed to create input id tensor: %w", err)
	}
	defer inputIds.Destroy()

	attMask, err := ort.NewTensor(ort.NewShape(encs.BatchSize, encs.SequenceLength), encs.FlattenedAttentionMasks)
	if err != nil {
		return nil, fmt.Errorf("failed to create attention mask tensor: %w", err)
	}
	defer attMask.Destroy()

	typeIds, err := ort.NewTensor(ort.NewShape(encs.BatchSize, encs.SequenceLength), encs.FlattenedTypeIds)
	if err != nil {
		return nil, fmt.Errorf("failed to create type id tensor: %w", err)
	}
	defer typeIds.Destroy()

	outputShape := ort.NewShape(encs.BatchSize, encs.SequenceLength, 384)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	session, err := getBatchSession()
	if err != nil {
		return nil, err
	}

	inputs := []ort.Value{inputIds, attMask, typeIds}
	outputs := []ort.Value{outputTensor}

	batchSessionMu.Lock()
	err = session.Run(inputs, outputs)
	batchSessionMu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("failed to run ONNX session: %w", err)
	}

	outputData := outputTensor.GetData()
	pooled := meanPoolBatch(outputData, int(encs.BatchSize), 384, int(encs.SequenceLength), encs.AttentionMasks)

	validated := make([]EmbeddingVector, encs.BatchSize)
	for i, p := range pooled {
		v, err := NewVector(p, 384)
		if err != nil {
			return nil, err
		}
		validated[i] = v
	}

	return validated, nil
}

func meanPoolSingle(rawOut []float32, hiddenSize int, seqLength int, attentionMask []int64) []float32 {
	split := make([][]float32, seqLength)
	for i := range seqLength {
		split[i] = rawOut[i*hiddenSize : (i+1)*hiddenSize]
	}

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
	return out
}

func meanPoolBatch(rawOut []float32, batchSize int, hiddenSize int, seqLength int, attentionMasks [][]int64) [][]float32 {
	vectorsRaw := make([][]float32, batchSize)

	for j := range batchSize {
		ofs := seqLength * hiddenSize
		vectorsRaw[j] = rawOut[j*ofs : (j+1)*ofs]
	}

	outs := make([][]float32, batchSize)
	for y := range batchSize {
		raw := vectorsRaw[y]
		split := make([][]float32, seqLength)
		for i := range seqLength {
			split[i] = raw[i*hiddenSize : (i+1)*hiddenSize]
		}

		r := make([]int, 0)
		for i := range attentionMasks[y] {
			if attentionMasks[y][i] == 1 {
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

		outs[y] = out
	}
	return outs
}

func getSingleSession() (*ort.DynamicAdvancedSession, error) {
	singleSessionInitOnce.Do(func() {
		session, err := newMiniLMSession()
		if err != nil {
			singleSessionErr = fmt.Errorf("failed to initialize single MiniLM session: %w", err)
			return
		}
		singleSession = session
	})
	if singleSessionErr != nil {
		return nil, singleSessionErr
	}
	if singleSession == nil {
		return nil, fmt.Errorf("single MiniLM session is unavailable")
	}
	return singleSession, nil
}

func getBatchSession() (*ort.DynamicAdvancedSession, error) {
	batchSessionInitOnce.Do(func() {
		session, err := newMiniLMSession()
		if err != nil {
			batchSessionErr = fmt.Errorf("failed to initialize batch MiniLM session: %w", err)
			return
		}
		batchSession = session
	})
	if batchSessionErr != nil {
		return nil, batchSessionErr
	}
	if batchSession == nil {
		return nil, fmt.Errorf("batch MiniLM session is unavailable")
	}
	return batchSession, nil
}

func newMiniLMSession() (*ort.DynamicAdvancedSession, error) {
	modelPath := resolveModelPath()
	session, err := ort.NewDynamicAdvancedSession(modelPath, modelInputNames, modelOutputNames, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create MiniLM session: %w", err)
	}
	return session, nil
}

func resolveModelPath() string {
	path := os.Getenv("MODEL_PATH")
	if path == "" {
		panic("model path not found, did config.Load() run?")
	}
	return path
}
