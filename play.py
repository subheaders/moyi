# play.py (improved)
import torch
import chess
import chess.pgn
import chess.engine
from datetime import datetime
from model import load_model
import torch.nn.functional as F
from utils.pgn_dataset import encode_board

# === Configuration ===
ENGINE_PATH = r"C:\Users\rukia\Desktop\mini-rl\maia\lc0.exe"
MAIA_WEIGHTS = r"C:\Users\rukia\Desktop\mini-rl\maia\maia-1100.pb.gz"
MODEL_PATH = "chess_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.8
USE_BFLOAT16 = True

# Board encoding is shared with training/pretraining and imported from utils.pgn_dataset

# === Select move from policy ===
def select_model_move(board, policy_logits):
    """
    Select a move using the same 4672-index move encoding as training.

    We reuse utils.pgn_dataset.move_to_index logic conceptually:
    - For each legal move, compute its 0..4671 index.
    - Use the corresponding log-prob from policy_logits.
    - Sample proportionally with temperature.

    To avoid circular imports, we implement a minimal move_to_index replica here.
    """
    from_sq_tensor = []
    idx_tensor = []
    legal_moves = list(board.legal_moves)
    scores = []

    for move in legal_moves:
        idx = _move_to_index_play(move, board)
        if idx is None:
            continue
        scores.append((move, policy_logits[0, idx].item()))

    if not scores:
        # Fallback: if mapping fails for some reason, pick a random legal move
        return legal_moves[0]

    moves, raw_scores = zip(*scores)
    probs = F.softmax(torch.tensor(raw_scores) / TEMPERATURE, dim=0)
    move = moves[torch.multinomial(probs, 1).item()]
    return move

def _move_to_index_play(move: chess.Move, board: chess.Board):
    """
    Local copy of move_to_index from utils.pgn_dataset to ensure consistency
    with the 4672-policy head layout used during pretraining.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    # Castling (2 planes)
    if board.is_castling(move):
        plane = 0 if chess.square_file(to_sq) > chess.square_file(from_sq) else 1
        return from_sq * 73 + plane

    # Promotions (16 planes)
    promotion = move.promotion
    if promotion is not None:
        df = chess.square_file(to_sq) - chess.square_file(from_sq)
        dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

        dir_idx = None
        if dr == 1 and df == 0:
            dir_idx = 0
        elif dr == 1 and df == 1:
            dir_idx = 1
        elif dr == 1 and df == -1:
            dir_idx = 2
        elif dr == 2 and df == 0:
            dir_idx = 3
        if dir_idx is None:
            return None

        promo_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        if promotion not in promo_pieces:
            return None
        piece_idx = promo_pieces.index(promotion)

        plane = 2 + dir_idx * 4 + piece_idx
        return from_sq * 73 + plane

    # Normal moves: 56 planes
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
        (1, 2), (2, 1), (-1, 2), (-2, 1),
        (1, -2), (2, -1), (-1, -2), (-2, -1),
    ]

    plane_offset = 18
    plane = None

    for d_idx, (dx, dy) in enumerate(directions):
        if d_idx < 8:
            for dist in range(1, 8):
                if df == dx * dist and dr == dy * dist:
                    plane = plane_offset + d_idx * 7 + (dist - 1)
                    break
            if plane is not None:
                break
        else:
            if df == dx and dr == dy:
                plane = plane_offset + 8 * 7 + (d_idx - 8)
                break

    if plane is None:
        return None

    idx = from_sq * 73 + (plane - plane_offset)
    if 0 <= idx < 4672:
        return idx
    return None


# === Play a Single Game ===
def play_game(model, engine):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers.update({
        "Event": "Model vs Maia Self-Play",
        "Site": "Localhost",
        "Date": datetime.now().strftime("%Y.%m.%d"),
        "Round": "1",
        "White": "RLModel",
        "Black": "Maia-1100",
        "Result": "*",
    })
    node = game

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float16):
                inp = encode_board(board).unsqueeze(0).to(DEVICE)
                policy, value = model(inp)
                move = select_model_move(board, policy)
        else:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move

        board.push(move)
        node = node.add_variation(move)
        node.comment = f"Eval: {float(value.item()):+.3f}" if board.turn == chess.BLACK else ""

    game.headers["Result"] = board.result()
    with open("game.pgn", "w", encoding="utf-8") as f:
        print(game, file=f)

    print(f"✅ Game finished: {board.result()} (PGN saved to game.pgn)")

# === Main Entry ===
def main():
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH, DEVICE)
    model.eval()

    print(f"Starting Maia engine from: {ENGINE_PATH}")
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    engine.configure({"WeightsFile": MAIA_WEIGHTS})

    play_game(model, engine)
    engine.quit()

if __name__ == "__main__":
    main()
