.PHONY: build run test clean setup setup-libs setup-models

# Project root directory
PROJECT_ROOT := $(shell pwd)

# Path to the built libtokenizers.a
LIBTOKENIZERS_PATH := $(PROJECT_ROOT)/libs/tokenizers
LIBTOKENIZERS_A := $(LIBTOKENIZERS_PATH)/libtokenizers.a

# MiniLM model paths
MODEL_DIR := $(PROJECT_ROOT)/models/all-MiniLM-L6-v2
TOKENIZER_JSON := $(MODEL_DIR)/tokenizer.json
TOKENIZER_CONFIG := $(MODEL_DIR)/tokenizer_config.json
MODEL_CONFIG := $(MODEL_DIR)/config.json
MODEL_ONNX := $(MODEL_DIR)/onnx/model.onnx
VOCAB_TXT := $(MODEL_DIR)/vocab.txt

# Set CGO_LDFLAGS to include the library path
export CGO_LDFLAGS := -L$(LIBTOKENIZERS_PATH)

# Export model paths as environment variables
export MODEL_DIR
export TOKENIZER_JSON
export TOKENIZER_CONFIG
export MODEL_CONFIG
export MODEL_ONNX
export VOCAB_TXT

build:
	@echo "Building with CGO_LDFLAGS=$(CGO_LDFLAGS)"
	@go build -ldflags="-extldflags '$(CGO_LDFLAGS)'" ./cmd/vect

run:
	@echo "Running with CGO_LDFLAGS=$(CGO_LDFLAGS)"
	@go run -ldflags="-extldflags '$(CGO_LDFLAGS)'" ./cmd/vect

test:
	@echo "Testing with CGO_LDFLAGS=$(CGO_LDFLAGS)"
	@go test -ldflags="-extldflags '$(CGO_LDFLAGS)'" ./...

clean:
	@go clean
	@echo "Cleaned build artifacts"

setup: setup-libs setup-models
	@echo ""
	@echo "=========================================="
	@echo "✓ Setup complete!"
	@echo "=========================================="
	@echo "Libraries: $(LIBTOKENIZERS_A)"
	@echo "Models: $(MODEL_DIR)"
	@echo ""

setup-libs: $(LIBTOKENIZERS_A)
	@echo "✓ Tokenizers library is ready"

$(LIBTOKENIZERS_A):
	@echo "Setting up tokenizers library..."
	@chmod +x scripts/setup_tokenizers.sh
	@bash scripts/setup_tokenizers.sh
	@if [ ! -f "$(LIBTOKENIZERS_A)" ]; then \
		echo "ERROR: Library file not found at $(LIBTOKENIZERS_A)"; \
		exit 1; \
	fi

setup-models: $(TOKENIZER_JSON) $(TOKENIZER_CONFIG) $(MODEL_CONFIG) $(MODEL_ONNX) $(VOCAB_TXT)
	@echo "✓ All model files are present"

$(TOKENIZER_JSON) $(TOKENIZER_CONFIG) $(MODEL_CONFIG) $(MODEL_ONNX) $(VOCAB_TXT):
	@echo "Setting up MiniLM embedding model..."
	@chmod +x scripts/setup_minilm.sh
	@bash scripts/setup_minilm.sh
	@echo "Verifying all required model files are present..."
	@for file in "$(TOKENIZER_JSON)" "$(TOKENIZER_CONFIG)" "$(MODEL_CONFIG)" "$(MODEL_ONNX)" "$(VOCAB_TXT)"; do \
		if [ ! -f "$$file" ]; then \
			echo "ERROR: Required file missing: $$file"; \
			exit 1; \
		fi; \
	done

