package embedding

import (
	"fmt"
	"log"
	"os"

	"github.com/daulet/tokenizers"
)

func load() {

}

func Encode(text string, strip bool) {
	tokenizerPath := os.Getenv("TOKENIZER_JSON")
	if tokenizerPath == "" {
		log.Fatal("failed getting tokenizer_json env variable")
	}
	tk, err := tokenizers.FromFile(tokenizerPath)

	if err != nil {
		log.Fatal("error tokenising from pretrained: ", err)
	}
	// release native resources
	defer tk.Close()

	if strip {
		// Use EncodeWithOptions to get attention mask for filtering padding
		encoding := tk.EncodeWithOptions(text, false, tokenizers.WithReturnTokens(), tokenizers.WithReturnAttentionMask())

		// Filter out padding tokens using attention mask
		var actualIDs []uint32
		var actualTokens []string
		for i, mask := range encoding.AttentionMask {
			if mask == 1 { // 1 means real token, 0 means padding
				actualIDs = append(actualIDs, encoding.IDs[i])
				actualTokens = append(actualTokens, encoding.Tokens[i])
			}
		}
		fmt.Printf("Total length (with padding): %d\n", len(encoding.IDs))
		fmt.Printf("Actual length (without padding): %d\n", len(actualIDs))
		fmt.Println("Ids (no padding): ", actualIDs)
		fmt.Println("Tokens (no padding): ", actualTokens)

		return
	} else {
		ids, tks := tk.Encode(text, false)
		fmt.Println("Ids: ", ids)
		fmt.Println("Tokens: ", tks)

		return
	}
}
