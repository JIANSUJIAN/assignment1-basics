import argparse
import os

import torch

from cs336_basics.generation import generate
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import load_checkpoint, load_config


def sample():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, default="final.pt")
    parser.add_argument("--vocab_path", type=str, default="output/ts_vocab.json")
    parser.add_argument("--merges_path", type=str, default="output/ts_merges.pkl")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps")

    args = parser.parse_args()
    print(f"Using device: {args.device}")

    config_path = os.path.join(args.checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    config = load_config(config_path)
    print(f"Loaded config: {config}")

    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"])

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=args.device,
    )
    model.to(args.device)

    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    optimizer = torch.optim.AdamW(model.parameters())  # Dummy optimizer
    iteration = load_checkpoint(ckpt_path, model, optimizer)
    print(f"Loaded weights from {ckpt_path} (Iteration {iteration})")

    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=args.device)
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    output_ids = generate(model, input_ids, args.max_new_tokens, args.temperature, args.top_p, eos_token_id=eos_id)

    output_text = tokenizer.decode(output_ids[0].tolist())
    print("\n--- Generated Text ---")
    print(output_text)


if __name__ == "__main__":
    sample()
