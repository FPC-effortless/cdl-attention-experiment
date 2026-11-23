"""Generate Tiny Shakespeare-style continuations from a trusted COG checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from data import CharTokenizer
from model import COGTinyLM, ModelConfig


def load_model(checkpoint: Path, device="cpu", threads=4):
    torch.set_num_threads(threads)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**payload["model_config"])
    model = COGTinyLM(config).to(device).eval()
    model.load_state_dict(payload["model"])
    if "tokenizer" in payload:
        tokenizer = CharTokenizer(payload["tokenizer"]["chars"])
    else:
        tokenizer = CharTokenizer.load(checkpoint.parent / "tokenizer.json")
    if config.vocab_size != len(tokenizer.chars):
        raise ValueError("Checkpoint and tokenizer vocabulary sizes differ")
    return model, tokenizer, payload


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=500, temperature=0.8,
             top_k=0, seed=42, device="cpu", prefill_chunk=256):
    if max_new_tokens < 0 or temperature < 0 or top_k < 0:
        raise ValueError("Length, temperature and top-k must be nonnegative")
    if prefill_chunk <= 0:
        raise ValueError("prefill_chunk must be positive")
    if not prompt:
        prompt = "\n"
    ids = tokenizer.encode(prompt)
    state = None
    logits = None
    tick = time.perf_counter()
    for begin in range(0, len(ids), prefill_chunk):
        x = torch.tensor([ids[begin:begin + prefill_chunk]], dtype=torch.long, device=device)
        logits, state = model(x, state)
    rng = torch.Generator(device=device).manual_seed(seed)
    generated = []
    for i in range(max_new_tokens):
        scores = logits[0, -1].float()
        if temperature == 0:
            nxt = scores.argmax().view(1)
        else:
            scores = scores / temperature
            if top_k:
                k = min(top_k, scores.numel())
                threshold = torch.topk(scores, k).values[-1]
                scores = scores.masked_fill(scores < threshold, float("-inf"))
            nxt = torch.multinomial(scores.softmax(-1), 1, generator=rng)
        token_id = int(nxt.item())
        generated.append(token_id)
        if i + 1 < max_new_tokens:
            logits, state = model(nxt.view(1, 1), state)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    return prompt + tokenizer.decode(generated), {
        "prompt_characters": len(ids), "generated_characters": len(generated),
        "seconds_including_prefill": time.perf_counter() - tick,
        "seed": seed, "temperature": temperature, "top_k": top_k,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="0 selects greedy deterministic decoding")
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k truncation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model, tokenizer, metadata = load_model(args.checkpoint, args.device, args.threads)
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,} parameters; checkpoint step {metadata.get('step', 'unknown')}.")
    if args.interactive:
        print("This is a Shakespeare-style text continuation model, not an instruction/chat model.")
        print(r"Enter a prompt; \n is accepted as a newline. Type /quit to exit.")
    while True:
        if args.interactive:
            try:
                prompt = input("\nprompt> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.strip() == "/quit":
                break
            prompt = prompt.replace("\\n", "\n")
        else:
            prompt = args.prompt
        try:
            text, stats = generate(model, tokenizer, prompt, args.max_new_tokens,
                                   args.temperature, args.top_k, args.seed, args.device)
        except ValueError as exc:
            if args.interactive:
                print(str(exc))
                continue
            raise
        print("\n" + text)
        print(f"\n[{stats['generated_characters']} new characters, {stats['seconds_including_prefill']:.2f}s including prompt]")
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        if not args.interactive:
            break


if __name__ == "__main__":
    main()

