"""Minimal CLI demo for running the TinyLlama chat model with GPT4All.

This script mirrors the behaviour of the Gradio demo but keeps things in a
simple command-line flow so it is easy to sanity-check that the local model is
able to respond.  Run it with ``python startLLM.py`` to see a short sample
interaction.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from gpt4all import GPT4All

MODEL_NAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = Path(__file__).resolve().parent / "models" / MODEL_NAME

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Place the model in the 'models' directory next to this script."
    )

llm = GPT4All(
    str(MODEL_PATH),
    allow_download=False,
    n_threads=max(2, (os.cpu_count() or 4) // 2),
    n_ctx=2048,
)


def ask_local_model(prompt: str, *, max_tokens: int = 120) -> str:
    """Generate a single response from the local TinyLlama model."""
    with llm.chat_session():
        return llm.generate(prompt, max_tokens=max_tokens, temp=0.1, top_p=0.95)


def main() -> None:
    question = (
        "Are these two user-entered addresses a match? "
        "Address1: 273 South Pelham Rd, Welland, L3C0E6. "
        "Address2: 273 S Pelham Rd, Welland, L3C0E6"
    )

    print("Loading model…")
    t0 = time.time()
    answer = ask_local_model(question)
    duration = time.time() - t0

    print("\nQuestion:\n" + question)
    print("\nModel answer:\n" + answer)
    print(f"\nCompleted in {duration:.1f}s")


if __name__ == "__main__":
    main()
