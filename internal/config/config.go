package config

import (
	"github.com/joho/godotenv"
)

func LoadConfig() (err error) {
	err = godotenv.Load()
	if err != nil {
		return err
	}
	return nil
}
