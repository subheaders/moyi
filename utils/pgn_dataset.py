import os
import io
import chess
import chess.pgn
import torch


def encode_board(board: chess.Board) -> torch.Tensor:
    """
    18x8x8 encoding matching train.py / pretrain.py:
    - 6 piece types x 2 colors = 12 planes
    - Remaining planes reserved for extensions; currently unused.
    """
    planes = torch.zeros((18, 8, 8), dtype=torch.float32)
    piece_map = board.piece_map()
    if not piece_map:
        return planes

    positions = []
    for idx, piece in piece_map.items():
        plane_offset = 0 if piece.color == chess.WHITE else 6
        piece_plane = piece.piece_type - 1
        row, col = divmod(idx, 8)
        positions.append((plane_offset + piece_plane, row, col))

    if positions:
        planes[tuple(zip(*positions))] = 1.0
    return planes


def move_to_index(move: chess.Move, board: chess.Board):
    """
    Map chess.Move to [0, 4672) index (64x73-style AlphaZero-like encoding).

    IMPORTANT:
    - Must match the policy head layout (model.py) and any decoding in play.py / RL.
    - If you already have a mapping elsewhere, replace this logic with that mapping.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    # Castling (2 planes)
    if board.is_castling(move):
        # 0: king-side, 1: queen-side
        plane = 0 if chess.square_file(to_sq) > chess.square_file(from_sq) else 1
        return from_sq * 73 + plane

    # Promotions (16 planes)
    promotion = move.promotion
    if promotion is not None:
        df = chess.square_file(to_sq) - chess.square_file(from_sq)
        dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

        dir_idx = None
        if dr == 1 and df == 0:
            dir_idx = 0   # forward
        elif dr == 1 and df == 1:
            dir_idx = 1   # up-right
        elif dr == 1 and df == -1:
            dir_idx = 2   # up-left
        elif dr == 2 and df == 0:
            dir_idx = 3   # double forward (placeholder)
        if dir_idx is None:
            return None

        promo_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        if promotion not in promo_pieces:
            return None
        piece_idx = promo_pieces.index(promotion)

        plane = 2 + dir_idx * 4 + piece_idx  # planes 2..17
        return from_sq * 73 + plane

    # Normal moves: 56 planes (sliding + knight-like)
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),      # rook-like
        (1, 1), (1, -1), (-1, 1), (-1, -1),    # bishop-like
        (1, 2), (2, 1), (-1, 2), (-2, 1),
        (1, -2), (2, -1), (-1, -2), (-2, -1),  # knight-like
    ]

    plane_offset = 18
    plane = None

    for d_idx, (dx, dy) in enumerate(directions):
        if d_idx < 8:
            # sliding
            for dist in range(1, 8):
                if df == dx * dist and dr == dy * dist:
                    plane = plane_offset + d_idx * 7 + (dist - 1)
                    break
            if plane is not None:
                break
        else:
            # knight-like
            if df == dx and dr == dy:
                plane = plane_offset + 8 * 7 + (d_idx - 8)
                break

    if plane is None:
        return None

    idx = from_sq * 73 + (plane - plane_offset)
    if 0 <= idx < 4672:
        return idx
    return None


def estimate_game_count(pgn_path: str, sample_bytes: int = 8_000_000) -> int | None:
    """
    Roughly estimate number of games in a large PGN by sampling.

    Strategy:
    - Read up to `sample_bytes` from the start.
    - Count occurrences of \"\\n\\n\" before a '[Event ' or similar that mark game headers.
    - Extrapolate based on file size.

    Returns:
        int: approximate number of games, or None if cannot estimate.
    """
    try:
        file_size = os.path.getsize(pgn_path)
        if file_size == 0:
            return None

        with open(pgn_path, "rb") as f:
            chunk = f.read(min(sample_bytes, file_size))

        # Decode best-effort
        text = chunk.decode("utf-8", errors="ignore")

        # Heuristic: count occurrences of \"\\n[Event \" as game headers
        header_token = "\n[Event "
        sample_games = text.count(header_token)
        if sample_games == 0:
            return None

        scale = file_size / len(text)
        est_total = int(sample_games * scale)

        # Ensure at least sample_games
        return max(est_total, sample_games)
    except Exception:
        return None


def iter_pgn_positions(pgn_path: str, max_games: int | None = None):
    """
    Stream (board, policy_one_hot, value) tuples from PGN.

    Uses encode_board + move_to_index; skips games without final result
    or moves that cannot be encoded.
    """
    with open(pgn_path, "r", encoding="utf-8") as f:
        game_idx = 0
        while True:
            if max_games is not None and game_idx >= max_games:
                break

            game = chess.pgn.read_game(f)
            if game is None:
                break

            result = game.headers.get("Result", "*")
            if result == "1-0":
                game_result = 1.0
            elif result == "0-1":
                game_result = -1.0
            elif result == "1/2-1/2":
                game_result = 0.0
            else:
                game_idx += 1
                continue

            board = game.board()
            node = game

            while node.variations:
                move = node.variation(0).move

                board_tensor = encode_board(board)

                move_index = move_to_index(move, board)
                if move_index is None:
                    board.push(move)
                    node = node.variation(0)
                    continue

                policy = torch.zeros(4672, dtype=torch.float32)
                policy[move_index] = 1.0

                stm = board.turn
                if game_result == 0.0:
                    value = 0.0
                elif (game_result == 1.0 and stm) or (game_result == -1.0 and not stm):
                    value = 1.0
                else:
                    value = -1.0

                yield board_tensor, policy, torch.tensor([value], dtype=torch.float32)

                board.push(move)
                node = node.variation(0)

            game_idx += 1
