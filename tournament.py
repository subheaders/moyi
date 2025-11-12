import os
import math
import random
import string
import csv
import torch
import chess
import chess.engine
import chess.pgn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from model import load_model
from utils.pgn_dataset import encode_board
from collections import defaultdict
from threading import Lock

# === Config ===
USE_BFLOAT16 = True
TEMPERATURE = 0.8
MODEL_PATH = "chess_model-2.pt"
GAMES_PER_OPPONENT = 50
THREADS = 5

# New flag: don't use Maia (lc0) on this system
USE_MAIA = False

# Paths - use system stockfish on Linux
STOCKFISH_PATH = "/usr/games/stockfish"
MAIA_PATH = None
MAIA_FOLDER = None

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"  
else:
    DEVICE = "cpu"

if DEVICE == 'mps':
    STOCKFISH_PATH = "./stockfish/stockfish"
    MAIA_PATH = './maia/lc0'
    MAIA_FOLDER = './maia'
else:
    pass


# === Opponent setups ===
OPPONENTS = []
 # Stockfish Elo levels: 800 to 2800 in 200-point intervals
for elo in range(1320, 2801, 200):
    OPPONENTS.append((f"stockfish-{elo}", STOCKFISH_PATH,
                      {"UCI_LimitStrength": True, "UCI_Elo": elo}, None))

# Only add Maia opponents when explicitly enabled
if USE_MAIA:
    maia_weights = {
        1100: "maia-1100.pb.gz",
        1200: "maia-1200.pb.gz",
        1300: "maia-1300.pb.gz",
        1400: "maia-1400.pb.gz",
        1500: "maia-1500.pb.gz",
        1600: "maia-1600.pb.gz",
        1800: "maia-1800.pb.gz",
        1900: "maia-1900.pb.gz",
    }
    for elo, fname in maia_weights.items():
        OPPONENTS.append((f"maia-{elo}", MAIA_PATH,
                          {"WeightsFile": os.path.join(MAIA_FOLDER, fname)}, fname))

# === Utility helpers ===
def random_id(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)

# === Move selection (same as before) ===
def select_model_move(board, policy_logits):
    legal_moves = list(board.legal_moves)
    scores = []
    for move in legal_moves:
        idx = _move_to_index_play(move, board)
        if idx is not None:
            scores.append((move, policy_logits[0, idx].item()))
    if not scores:
        return random.choice(legal_moves)
    moves, raw_scores = zip(*scores)
    probs = F.softmax(torch.tensor(raw_scores) / TEMPERATURE, dim=0)
    return moves[torch.multinomial(probs, 1).item()]

def _move_to_index_play(move, board):
    from_sq = move.from_square
    to_sq = move.to_square
    if board.is_castling(move):
        plane = 0 if chess.square_file(to_sq) > chess.square_file(from_sq) else 1
        return from_sq * 73 + plane
    promotion = move.promotion
    if promotion is not None:
        df = chess.square_file(to_sq) - chess.square_file(from_sq)
        dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)
        dirs = {(0,1):0,(1,1):1,(-1,1):2,(0,2):3}
        if (df, dr) not in dirs: return None
        promo_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        if promotion not in promo_pieces: return None
        plane = 2 + dirs[(df, dr)] * 4 + promo_pieces.index(promotion)
        return from_sq * 73 + plane
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),
                  (1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]
    plane_offset = 18
    plane = None
    for d_idx, (dx, dy) in enumerate(directions):
        if d_idx < 8:
            for dist in range(1,8):
                if df == dx*dist and dr == dy*dist:
                    plane = plane_offset + d_idx*7 + (dist-1)
                    break
        else:
            if df == dx and dr == dy:
                plane = plane_offset + 8*7 + (d_idx-8)
                break
    if plane is None: return None
    idx = from_sq*73 + (plane-plane_offset)
    return idx if 0 <= idx < 4672 else None

# Global lock for model inference to avoid GPU context contention across threads
MODEL_LOCK = Lock()

# === Play Game ===
def play_single_game(model, engine_path, engine_config, opponent_name):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure(engine_config)
    board = chess.Board()
    as_white = random.choice([True, False])
    game = chess.pgn.Game()
    game.headers.update({
        "Event": f"Model vs {opponent_name}",
        "Site": "Localhost",
        "Date": datetime.now().strftime("%Y.%m.%d"),
        "White": "RLModel" if as_white else opponent_name,
        "Black": opponent_name if as_white else "RLModel",
        "Result": "*"
    })
    node = game

    while not board.is_game_over():
        if (board.turn == chess.WHITE and as_white) or (board.turn == chess.BLACK and not as_white):
            # Use a lock around model inference to safely share a single GPU model across threads.
            with MODEL_LOCK, torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float16):
                inp = encode_board(board).unsqueeze(0).to(DEVICE, non_blocking=True)
                policy, value = model(inp)
                move = select_model_move(board, policy)
        else:
            # Let each thread use its own engine with a short time limit
            result_info = engine.play(board, chess.engine.Limit(time=0.05))
            move = result_info.move
        board.push(move)
        node = node.add_variation(move)

    result = board.result()
    game.headers["Result"] = result
    engine.quit()

    # Save PGN
    folder = os.path.join("tournament", opponent_name)
    ensure_folder(folder)
    filename = f"result_{random_id()}.pgn"
    with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
        print(game, file=f)
    return result, game.headers["White"], game.headers["Black"]

# === Tournament Runner ===
def run_tournament():
    # Load model once and share across threads; pin to eval and inference dtype
    model = load_model(MODEL_PATH, DEVICE).eval()
    ensure_folder("tournament")
    stats = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "games": 0})

    for name, path, config, _ in OPPONENTS:
        print(f"\n=== Playing vs {name} ===")
        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = [executor.submit(play_single_game, model, path, config, name) for _ in range(GAMES_PER_OPPONENT)]
            for f in tqdm(as_completed(futures), total=GAMES_PER_OPPONENT, desc=f"{name}"):
                try:
                    result, white, black = f.result()
                    stats[name]["games"] += 1
                    if result == "1/2-1/2":
                        stats[name]["draws"] += 1
                    elif (result == "1-0" and white == "RLModel") or (result == "0-1" and black == "RLModel"):
                        stats[name]["wins"] += 1
                    else:
                        stats[name]["losses"] += 1
                except Exception as e:
                    print(f"Error: {e}")
    return stats

# === Elo estimation ===
def estimate_elo(stats):
    anchors = {"stockfish-1600": 1600, "stockfish-2000": 2000,
               "stockfish-2400": 2400, "stockfish-2800": 2800}
    pairs = []
    for opp, data in stats.items():
        if opp in anchors and data["games"] > 0:
            score = (data["wins"] + 0.5 * data["draws"]) / data["games"]
            pairs.append((anchors[opp], score))
    if not pairs:
        return None
    # Weighted average Elo diff
    avg_diff = 0
    total_w = 0
    for opp_elo, score in pairs:
        if score <= 0:
            diff = 600
        elif score >= 1:
            diff = -600
        else:
            diff = 400 * math.log10((1 / score) - 1)
        avg_diff += (opp_elo - diff)
        total_w += 1
    return avg_diff / total_w if total_w > 0 else None

# === Save results ===
def save_results(stats, est_elo):
    csv_path = os.path.join("tournament", "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Opponent", "Games", "Wins", "Draws", "Losses", "Score%", "EstimatedElo"])
        for name, data in stats.items():
            score = (data["wins"] + 0.5 * data["draws"]) / data["games"] * 100 if data["games"] else 0
            writer.writerow([name, data["games"], data["wins"], data["draws"],
                             data["losses"], f"{score:.1f}", f"{est_elo:.1f}" if est_elo else ""])
    print(f"\n✅ Saved corrected results to: {csv_path}")

# === Main ===
if __name__ == "__main__":
    results = run_tournament()
    print("\n=== Corrected Results ===")
    for name, d in results.items():
        score = (d["wins"] + 0.5 * d["draws"]) / d["games"] * 100 if d["games"] else 0
        print(f"{name}: {score:.1f}%  ({d['wins']}W/{d['draws']}D/{d['losses']}L)")
    est = estimate_elo(results)
    if est:
        print(f"\n🎯 Estimated model Elo ≈ {est:.0f}")
    save_results(results, est)
