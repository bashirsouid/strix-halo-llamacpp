#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 ./server.py serve nemotron-nano-q4 --backend rocm

