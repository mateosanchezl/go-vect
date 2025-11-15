package config

import (
	"log"
	"os"
	"path/filepath"

	"github.com/joho/godotenv"
)

func LoadConfig() (err error) {
	initEnv()

	if _, err := os.Stat(".env"); os.IsNotExist(err) {
		log.Println("warning: .env file not found. make sure to set this up to use huggingface embedding models via api")
		return nil
	}

	err = godotenv.Load()
	if err != nil {
		return err
	}
	return nil
}

func initEnv() {
	tokenizerPath := os.Getenv("TOKENIZER_JSON")
	if tokenizerPath == "" {
		tokenizerPath = "models/all-MiniLM-L6-v2/tokenizer.json"
	}

	if !filepath.IsAbs(tokenizerPath) {
		wd, err := os.Getwd()
		if err == nil {
			tokenizerPath = filepath.Join(wd, tokenizerPath)
		}
	}

	os.Setenv("TOKENIZER_JSON", tokenizerPath)
}
