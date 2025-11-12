# train_transformer.py
import os
import argparse
import torch
import torch.optim as optim
import gc
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader

# Use transformer_model loader
from transformer_model import load_model
from utils.pgn_dataset import iter_pgn_positions, estimate_game_count



# ==============================
# Optional performance toggles
# ==============================
USE_TORCH_COMPILE = False  # Enable torch.compile()
USE_FP8 = False

if USE_FP8:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.fp8 import fp8_autocast, fp8_update
else:
    te = None  # type: ignore

            # Enable FP8 training (PyTorch native autocast)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

DATA_PATH = os.path.join("data", "lichess_elite_2023-01.pgn")

if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    except Exception:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
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
        worker_info = None
        try:
            from torch.utils.data import get_worker_info
            worker_info = get_worker_info()
        except Exception:
            worker_info = None

        if worker_info is None:
            return iter_pgn_positions(self.pgn_path, max_games=self.max_games)
        else:
            return iter_pgn_positions(
                self.pgn_path,
                max_games=self.max_games,
                worker_id=worker_info.id,
                num_workers=worker_info.num_workers,
            )

# ==============================
# Training logic
# ==============================
def pretrain(
    pgn_path: str = DATA_PATH,
    model_path_out: str = "chess_transformer.pt",
    init_model_path: str | None = None,
    max_games: int | None = None,
    epochs: int = 1,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_policy: float = 1.0,
    weight_value: float = 1.0,
    # hyperparameters for transformer (defaults can be tuned)
    num_layers: int = 8,
    dim: int = 512,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    attn_dropout: float = 0.0,
    use_rope: bool = True,
    share_heads: bool = True,
):
    assert os.path.exists(pgn_path), f"PGN file not found: {pgn_path}"

    # Pass config kwargs to the transformer loader
    model = load_model(
        init_model_path,
        DEVICE,
        num_layers=num_layers,
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attn_dropout=attn_dropout,
        use_rope=use_rope,
        share_heads=share_heads,
    )

    # Log total parameter count at startup (redundant with model print, but helpful)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: total={total_params:,} | trainable={trainable_params:,}")

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
    if DEVICE == "cuda":
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
        )

    try:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            total_games = None
            if est_games is not None:
                effective_games = est_games if max_games is None else min(est_games, max_games)
                total_games = effective_games

            pbar = tqdm(
                desc=f"Pretraining games (epoch {epoch + 1}/{epochs})",
                unit="games",
                total=total_games,
            )

            for batch_boards, batch_policy, batch_value, batch_is_last in dataloader:
                loss = train_batch(
                    model,
                    optimizer,
                    batch_boards,
                    batch_policy,
                    batch_value,
                    weight_policy,
                    weight_value,
                )
                try:
                    games_completed = int(batch_is_last.sum().item())
                except Exception:
                    games_completed = int(sum(int(x) for x in batch_is_last))

                if games_completed > 0:
                    pbar.update(games_completed)

                pbar.set_postfix({"loss": f"{loss:.4f}"})

            pbar.close()
            print(f"Epoch {epoch + 1} completed.")

            torch.save(model.state_dict(), model_path_out)
            print(f"Saved pretrained model to {model_path_out}")
    finally:
        try:
            del dataloader
        except Exception:
            pass
        gc.collect()
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    print("Pretraining complete.")

# ==============================
# Single batch training
# ==============================
def train_batch(model, optimizer, boards, policies, values, weight_policy, weight_value):
    model.train()
    X = boards.to(DEVICE, non_blocking=True)
    target_p = policies.to(DEVICE, non_blocking=True)
    target_v = values.to(DEVICE, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    if USE_FP8 and DEVICE == "cuda":
        with fp8_autocast(enabled=True):
            log_p, v = model(X)
            policy_loss = -(target_p * log_p).sum(dim=1).mean()
            value_loss = torch.mean((v - target_v) ** 2)
            loss = weight_policy * policy_loss + weight_value * value_loss
    else:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            log_p, v = model(X)
            policy_loss = -(target_p * log_p).sum(dim=1).mean()
            value_loss = torch.mean((v - target_v) ** 2)
            loss = weight_policy * policy_loss + weight_value * value_loss

    loss.backward()
    if USE_FP8:
        fp8_update(model)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())
# ==============================
# CLI parsing
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="Supervised pretraining from PGN (Transformer).")
    parser.add_argument("--pgn", type=str, default=DATA_PATH, help="Path to PGN file.")
    parser.add_argument("--init", type=str, default=None, help="Optional initial model checkpoint.")
    parser.add_argument("--out", type=str, default="chess_transformer.pt", help="Output model path.")
    parser.add_argument("--max-games", type=int, default=None, help="Limit number of games.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of passes over PGN.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-policy", type=float, default=1.0, help="Policy loss weight.")
    parser.add_argument("--weight-value", type=float, default=1.0, help="Value loss weight.")
    # transformer hyperparams exposed as CLI options
    parser.add_argument("--num-layers", type=int, default=8, help="Transformer layers")
    parser.add_argument("--dim", type=int, default=256, help="Model width")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="MLP expansion ratio")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--attn-dropout", type=float, default=0.0, help="Attention dropout")
    parser.add_argument("--use-rope", action="store_true", help="Enable RoPE")
    parser.add_argument("--share-heads", action="store_true", help="Share head projection for policy/value")
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
        num_layers=args.num_layers,
        dim=args.dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        use_rope=args.use_rope,
        share_heads=args.share_heads,
    )

if __name__ == "__main__":
    main()
