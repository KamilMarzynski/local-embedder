import argparse
import json
import sys
import requests

DEFAULT_URL = "http://localhost:11434/api/embed"
DEFAULT_MODEL = "mxbai-embed-large"

def ollama_embed(texts, model=DEFAULT_MODEL, url=DEFAULT_URL, truncate=True):
    payload = {
        "model": model,
        "input": texts,
        "truncate": truncate,
    }

    r = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    return r.json()

def main():
    p = argparse.ArgumentParser(description="Generate embeddings locally using Ollama /api/embed.")
    p.add_argument("--text", "-t", help="Text to embed. If omitted, reads from stdin.")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    p.add_argument("--url", default=DEFAULT_URL, help=f"Ollama embed endpoint (default: {DEFAULT_URL})")
    p.add_argument("--out", "-o", help="Optional output file (JSON). If omitted, prints to stdout.")
    args = p.parse_args()

    text = args.text if args.text is not None else sys.stdin.read().strip()
    if not text:
        raise SystemExit("Empty input. Pass --text or provide stdin.")

    # Używamy listy, żebyś łatwo rozszerzył to na batch
    res = ollama_embed([text], model=args.model, url=args.url)

    output = {
        "model": args.model,
        "text": text,
        "response": res,
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False)
    else:
        print(json.dumps(output, ensure_ascii=False))

if __name__ == "__main__":
    main()
