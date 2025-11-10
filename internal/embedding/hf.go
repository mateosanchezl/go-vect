package embedding

type HfEmbeddingRequest struct {
	Inputs []string `json:"inputs"`
}
type HfEmbeddingModel struct {
	Token    string
	ModelUrl string
}

func (hf *HfEmbeddingModel) Embed(chunks []string) (embeddings []EmbeddingVector, err error) {
	err = godotenv.Load()
	if err != nil {
		return nil, err
	}

	payload := HfEmbeddingRequest{
		Inputs: chunks,
	}
	jsonData, err := json.Marshal(payload)

	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", hf.ModelUrl, bytes.NewBuffer(jsonData))
	req.Header.Set("Authorization", "Bearer "+hf.Token)
	req.Header.Set("Content-Type", "application/json")
	if err != nil {
		return nil, err
	}

	client := http.DefaultClient

	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	body, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}

	var vect []float64
	json.Unmarshal(body, &vect)

	validated, err := NewVector(vect)
	if err != nil {
		return nil, err
	}

	vs := []EmbeddingVector{validated}

	return vs, nil
}
