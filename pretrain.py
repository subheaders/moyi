import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import chess
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader

from model import load_model
from utils.pgn_dataset import iter_pgn_positions, estimate_game_count

# ==============================
# Optional performance toggles
# ==============================
USE_TORCH_COMPILE = False  # Enable torch.compile()
USE_FP8 = False            # Enable FP8 training (PyTorch native autocast)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.path.join("data", "elite_games.pgn")

if DEVICE == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# ==============================
# Dataset wrapper for DataLoader
# ==============================
class PGNDataset(IterableDataset):
    def __init__(self, pgn_path, max_games=None):
        self.pgn_path = pgn_path
        self.max_games = max_games

    def __iter__(self):
        return iter_pgn_positions(self.pgn_path, max_games=self.max_games)

# ==============================
# Training logic
# ==============================
def pretrain(
    pgn_path: str = DATA_PATH,
    model_path_out: str = "chess_model-large.pt",
    init_model_path: str | None = None,
    max_games: int | None = None,
    epochs: int = 1,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_policy: float = 1.0,
    weight_value: float = 1.0,
):
    assert os.path.exists(pgn_path), f"PGN file not found: {pgn_path}"

    model = load_model(init_model_path, DEVICE)
    model.train()

    # Optional torch.compile
    if USE_TORCH_COMPILE:
        print("Compiling model with torch.compile() for performance...")
        model = torch.compile(model)

    # Use fused AdamW when available
    use_fused = DEVICE == "cuda"
    try:
        optimizer = optim.AdamW(model.parameters(), lr=lr, fused=use_fused)
    except TypeError:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    if DEVICE == "cuda":
        if not torch.cuda.is_bf16_supported() and not USE_FP8:
            raise RuntimeError("bf16 required but not supported on this CUDA device.")
        print(f"Using device: {DEVICE}")
        print(f"FP8 training: {'enabled' if USE_FP8 else 'disabled (using bf16)'}")
    else:
        print("WARNING: Running on CPU; training will be fp32.")

    print(f"Pretraining from PGN: {pgn_path}")
    if max_games:
        print(f"Max games: {max_games}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")

    est_games = estimate_game_count(pgn_path)
    if est_games is not None:
        print(f"Estimated total games in PGN: {est_games}")
    else:
        print("Could not estimate total games; tqdm will show dynamic count only.")

    dataset = PGNDataset(pgn_path, max_games=max_games)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        total_positions = None
        if est_games is not None:
            effective_games = est_games if max_games is None else min(est_games, max_games)
            total_positions = effective_games * 80

        pbar = tqdm(
            desc=f"Pretraining positions (epoch {epoch + 1}/{epochs})",
            unit="pos",
            total=total_positions,
        )

        for batch_boards, batch_policy, batch_value in dataloader:
            loss = train_batch(
                model,
                optimizer,
                batch_boards,
                batch_policy,
                batch_value,
                weight_policy,
                weight_value,
            )
            pbar.update(len(batch_boards))
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        pbar.close()
        print(f"Epoch {epoch + 1} completed.")

        torch.save(model.state_dict(), model_path_out)
        print(f"Saved pretrained model to {model_path_out}")

    print("Pretraining complete.")

# ==============================
# Single batch training
# ==============================
def train_batch(
    model,
    optimizer,
    boards,
    policies,
    values,
    weight_policy: float,
    weight_value: float,
):
    model.train()

    X = boards.to(DEVICE, non_blocking=True)
    target_p = policies.to(DEVICE, non_blocking=True)
    target_v = values.to(DEVICE, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    # FP8 / bf16 autocast logic
    if USE_FP8 and DEVICE == "cuda" and hasattr(torch, "float8_e4m3fn"):
        with torch.autocast(device_type="cuda", dtype=torch.float8_e4m3fn):
            log_p, v = model(X)
            policy_loss = -(target_p * log_p).sum(dim=1).mean()
            value_loss = torch.mean((v - target_v) ** 2)
            loss = weight_policy * policy_loss + weight_value * value_loss
    else:
        autocast_enabled = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
        with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=torch.bfloat16):
            log_p, v = model(X)
            policy_loss = -(target_p * log_p).sum(dim=1).mean()
            value_loss = torch.mean((v - target_v) ** 2)
            loss = weight_policy * policy_loss + weight_value * value_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return float(loss.item())

# ==============================
# CLI parsing
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="Supervised pretraining from PGN.")
    parser.add_argument("--pgn", type=str, default=DATA_PATH, help="Path to PGN file.")
    parser.add_argument("--init", type=str, default=None, help="Optional initial model checkpoint.")
    parser.add_argument("--out", type=str, default="chess_model-large.pt", help="Output model path.")
    parser.add_argument("--max-games", type=int, default=None, help="Limit number of games.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of passes over PGN.")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-policy", type=float, default=1.0, help="Policy loss weight.")
    parser.add_argument("--weight-value", type=float, default=1.0, help="Value loss weight.")
    return parser.parse_args()

def main():
    args = parse_args()
    pretrain(
        pgn_path=args.pgn,
        model_path_out=args.out,
        init_model_path=args.init,
        max_games=args.max_games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_policy=args.weight_policy,
        weight_value=args.weight_value,
    )

if __name__ == "__main__":
    main()
