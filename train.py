import os
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import numpy as np
import concurrent.futures
from threading import Lock
from model import load_model
from tqdm import tqdm

# --- Configuration ---
USE_COMPILE = False
USE_BFLOAT16 = True       # use bfloat16 (recommended on RTX 4060)
USE_AMP = False           # GradScaler only for fp16; disabled for bf16
USE_CUDA_GRAPHS = False   # Disabled by default (unstable on some setups)
ASYNC_WORKERS = 4
BATCH_SELF_PLAY = True
GRAD_ACCUMULATION_STEPS = 4
CUDA_GRAPH_CAPTURE_BATCH = 64

# --- Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def setup_optimizations():
    if DEVICE == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)
        except Exception:
            pass
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))

setup_optimizations()

# --- Board Encoding ---
def encode_board(board):
    planes = torch.zeros((18, 8, 8), dtype=torch.float32)
    piece_map = board.piece_map()
    positions = []
    for idx, piece in piece_map.items():
        plane_offset = 0 if piece.color == chess.WHITE else 6
        piece_plane = piece.piece_type - 1
        row, col = divmod(idx, 8)
        positions.append((plane_offset + piece_plane, row, col))
    if positions:
        planes[tuple(zip(*positions))] = 1.0
    return planes

def calculate_reward(outcome):
    if outcome is None:
        return 0
    if outcome.winner is True:
        return 1
    if outcome.winner is False:
        return -1
    return 0

# --- Self-Play (batch) ---
@torch.no_grad()
def self_play_batch(model, num_games=50, max_moves=100):
    model.eval()
    games = [chess.Board() for _ in range(num_games)]
    active_games = list(range(num_games))
    all_states, all_rewards = [], []
    game_states = {i: [] for i in range(num_games)}

    move_count = 0
    while active_games and move_count < max_moves:
        batch_states, game_indices = [], []
        for game_idx in active_games:
            board = games[game_idx]
            if not board.is_game_over():
                state = encode_board(board)
                batch_states.append(state)
                game_indices.append(game_idx)
                game_states[game_idx].append(state)
        if not batch_states:
            break

        state_batch = torch.stack(batch_states).to(DEVICE)
        policy_batch, _ = model(state_batch)

        new_active_games = []
        for i, game_idx in enumerate(game_indices):
            board = games[game_idx]
            legal_moves = list(board.legal_moves)
            if legal_moves and not board.is_game_over():
                move_idx = torch.argmax(policy_batch[i]).item() % len(legal_moves)
                chosen_move = legal_moves[move_idx]
                board.push(chosen_move)
                if not board.is_game_over():
                    new_active_games.append(game_idx)
                else:
                    outcome = board.outcome()
                    reward = calculate_reward(outcome)
                    all_states.extend(game_states[game_idx])
                    all_rewards.extend([reward] * len(game_states[game_idx]))
        active_games = new_active_games
        move_count += 1

    # incomplete games => treat as draw
    for game_idx in active_games:
        if game_states[game_idx]:
            all_states.extend(game_states[game_idx])
            all_rewards.extend([0] * len(game_states[game_idx]))

    return all_states, all_rewards

# --- Async Data Generator ---
class AsyncDataGenerator:
    def __init__(self, model, num_workers=4):
        self.model = model
        self.num_workers = num_workers
        self.data_buffer = []
        self.lock = Lock()

    def generate_async(self, total_games):
        self.data_buffer.clear()
        games_per_worker = max(1, total_games // self.num_workers)
        remaining_games = total_games % self.num_workers

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(self.num_workers):
                games_to_play = games_per_worker + (1 if i < remaining_games else 0)
                if games_to_play > 0:
                    futures.append(
                        executor.submit(self_play_batch, self.model, games_to_play)
                    )

            for future in concurrent.futures.as_completed(futures):
                try:
                    states, rewards = future.result()
                    with self.lock:
                        self.data_buffer.extend(list(zip(states, rewards)))
                except Exception as e:
                    print("Error in async generation:", e)

        return self.data_buffer

# --- Training ---
def main():
    model_path = "chess_model.pt"
    model = load_model(None, DEVICE)

    # 🔥 Load saved model if available
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded existing model from {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load {model_path}: {e}")
    else:
        print("🆕 No saved model found — starting fresh.")

    if USE_COMPILE:
        try:
            model = torch.compile(model)
        except Exception as e:
            print("Compile failed:", e)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model parameter count: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, fused=(DEVICE == "cuda"), foreach=False)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=100, steps_per_epoch=200, pct_start=0.1
    )

    EPOCHS = 500
    GAMES_PER_EPOCH = 200  # reduced to avoid OOM
    total_games = EPOCHS * GAMES_PER_EPOCH

    data_gen = AsyncDataGenerator(model, num_workers=ASYNC_WORKERS) if ASYNC_WORKERS > 1 else None

    pbar = tqdm(total=total_games, desc="Training Games", ncols=100)

    cuda_graph = None
    precision_dtype = torch.bfloat16 if USE_BFLOAT16 else torch.float16

    for epoch in range(EPOCHS):
        model.train()

        # --- Data Generation ---
        if data_gen and BATCH_SELF_PLAY:
            game_data = data_gen.generate_async(GAMES_PER_EPOCH)
            pbar.update(GAMES_PER_EPOCH)
        else:
            game_data = []
            for _ in range(GAMES_PER_EPOCH):
                states, rewards = self_play_batch(model)
                game_data.extend(zip(states, rewards))
                pbar.update(1)

        if not game_data:
            continue

        states, rewards = zip(*game_data)
        X = torch.stack(states).to(DEVICE, non_blocking=True)
        y = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batch_size = max(1, len(X) // GRAD_ACCUMULATION_STEPS)

        for i in range(0, len(X), batch_size):
            end_idx = min(i + batch_size, len(X))
            batch_X = X[i:end_idx]
            batch_y = y[i:end_idx]

            with torch.autocast(device_type=DEVICE, dtype=precision_dtype):
                pred_p, pred_v = model(batch_X)
                value_loss = torch.mean((pred_v - batch_y) ** 2)
                target_policy = torch.ones_like(pred_p) / pred_p.size(1)
                policy_loss = -torch.sum(target_policy * torch.log_softmax(pred_p, dim=1)) / pred_p.size(0)
                loss = (value_loss + 0.01 * policy_loss) / GRAD_ACCUMULATION_STEPS

            loss.backward()
            total_loss += float(loss.item() * GRAD_ACCUMULATION_STEPS)

            step_idx = (i // batch_size) + 1
            if (step_idx % GRAD_ACCUMULATION_STEPS) == 0 or (end_idx == len(X)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        pbar.set_postfix({
            "loss": f"{total_loss:.4f}",
            "epoch": f"{epoch + 1}/{EPOCHS}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

        # 🔄 Save model after every epoch
        torch.save(model.state_dict(), model_path)

    pbar.close()
    print("\n✅ Training complete — model saved as chess_model.pt")

if __name__ == "__main__":
    main()
