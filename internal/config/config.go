package config

import (
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/joho/godotenv"
)

var (
	ortInitOnce sync.Once
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
	initTokenizerPath()
	initOnnxRuntime()
}

func initTokenizerPath() {
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

func initOnnxRuntime() {
	initOnnxLibPath() // Ensure we have a path in env
	path := os.Getenv("ONNX_RUNTIME_LIB_PATH")
	ort.SetSharedLibraryPath(path)

	ortInitOnce.Do(func() { // Initialise once
		err := ort.InitializeEnvironment()
		if err != nil {
			log.Fatalln("failed loading onnx runtime: ", err)
		}
	})
}

func initOnnxLibPath() {
	onnxEnvPath := os.Getenv("ONNX_RUNTIME_LIB_PATH")
	if onnxEnvPath == "" {
		onnxPath := getDefaultOnnxLibPath()
		if onnxPath == "" {
			log.Fatalln("error: no onnx runtime path was found in env or where it is commonly located on your system type. please set this up in your .env")
		}
		os.Setenv("ONNX_RUNTIME_LIB_PATH", onnxPath)
	}
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
