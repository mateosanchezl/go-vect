package config

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/joho/godotenv"
)

var (
	ortInitOnce sync.Once
	tkInitOnce  sync.Once
	Tokenizer   *tokenizer.Tokenizer // For global use
)

func LoadConfig() (err error) {
	err = initEnv()
	if err != nil {
		return fmt.Errorf("failed to initialize environment: %w", err)
	}

	if _, err := os.Stat(".env"); os.IsNotExist(err) {
		log.Println("warning: .env file not found. make sure to set this up to use huggingface embedding models via api")
		return nil
	}

	err = godotenv.Load()
	if err != nil {
		return fmt.Errorf("failed loading env: %w", err)
	}
	return nil
}

func initEnv() error {
	err := initTokenizer()
	if err != nil {
		return fmt.Errorf("failed to initialize tokenizer: %w", err)
	}

	err = initOnnxRuntime()
	if err != nil {
		return fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	return nil
}

func initTokenizer() error {
	var initErr error
	tkInitOnce.Do(func() {
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

		tk, err := pretrained.FromFile(tokenizerPath)
		if err != nil {
			initErr = fmt.Errorf("failed to initialize tokenizer: %w", err)
			return
		}

		// Values from the MiniLM tokeniser config
		tk.WithTruncation(&tokenizer.TruncationParams{MaxLength: 384, Strategy: tokenizer.LongestFirst})
		tk.WithPadding(&tokenizer.PaddingParams{
			Strategy:  *tokenizer.NewPaddingStrategy(tokenizer.WithFixed(128)),
			Direction: tokenizer.Right,
			PadId:     0,
			PadTypeId: 0,
			PadToken:  "[PAD]",
		})

		Tokenizer = tk
		fmt.Println("tokenizer initialised")
	})
	return initErr
}

func initOnnxRuntime() error {
	err := initOnnxLibPath() // Ensure we have a path in env
	if err != nil {
		return fmt.Errorf("failed to initialize ONNX library path: %w", err)
	}

	path := os.Getenv("ONNX_RUNTIME_LIB_PATH")
	ort.SetSharedLibraryPath(path)

	var initErr error
	ortInitOnce.Do(func() { // Initialise once
		err := ort.InitializeEnvironment()
		if err != nil {
			initErr = fmt.Errorf("failed to initialize ONNX runtime: %w", err)
			return
		}
		fmt.Println("onnx runtime initialised")
	})
	return initErr
}

func initOnnxLibPath() error {
	onnxEnvPath := os.Getenv("ONNX_RUNTIME_LIB_PATH")
	if onnxEnvPath == "" {
		onnxPath := getDefaultOnnxLibPath()
		if onnxPath == "" {
			return errors.New("no onnx runtime path was found in env or where it is commonly located on your system type. please set this up in your .env")
		}
		err := os.Setenv("ONNX_RUNTIME_LIB_PATH", onnxPath)
		if err != nil {
			return fmt.Errorf("failed to set onnx path environment variable: %w", err)
		}
	}
	return nil
}

func getDefaultOnnxLibPath() string {
	// Try common installation paths based on platform
	switch runtime.GOOS {
	case "darwin":
		paths := []string{
			"/opt/homebrew/lib/libonnxruntime.dylib", // Apple Silicon
			"/usr/local/lib/libonnxruntime.dylib",    // Intel Mac
		}
		for _, p := range paths {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	case "linux":
		paths := []string{
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/libonnxruntime.so",
		}
		for _, p := range paths {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	case "windows":
		paths := []string{
			"C:\\Program Files\\onnxruntime\\lib\\onnxruntime.dll",
			"C:\\onnxruntime\\lib\\onnxruntime.dll",
		}
		for _, p := range paths {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}

	return ""
}
