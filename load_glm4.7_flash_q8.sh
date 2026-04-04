#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 ./server.py serve glm-4.7-flash-q8 --backend rocm
