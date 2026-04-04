#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 ./server.py serve qwen3.5-35b-q8 --backend rocm

