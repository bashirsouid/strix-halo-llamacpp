#!/usr/bin/env bash

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-4-119b",
    "messages": [{"role": "user", "content": "Write a haiku about speculative decoding"}],
    "stream": false
  }' | jq .
  