.PHONY: build run test clean setup help

# Model paths
MODEL_DIR := models/all-MiniLM-L6-v2
MODEL_FILES := $(MODEL_DIR)/tokenizer.json $(MODEL_DIR)/tokenizer_config.json \
               $(MODEL_DIR)/config.json $(MODEL_DIR)/onnx/model.onnx $(MODEL_DIR)/vocab.txt

build:
	@go build -o bin/vect ./cmd/vect

run: setup
	@go run ./cmd/vect

test:
	@go test ./...

clean:
	@rm -rf bin/
	@go clean

setup: $(MODEL_FILES)
	@echo "✓ Setup complete"

$(MODEL_FILES):
	@echo "Downloading model files..."
	@chmod +x scripts/setup_minilm.sh
	@bash scripts/setup_minilm.sh

help:
	@echo "make setup  - Download model files"
	@echo "make build  - Build binary"
	@echo "make run    - Run application"
	@echo "make test   - Run tests"
	@echo "make clean  - Clean artifacts"