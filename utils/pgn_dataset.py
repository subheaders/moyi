import torch, chess, chess.pgn

def encode_board(b:chess.Board)->torch.Tensor:
    p=torch.zeros(18,8,8); m=b.piece_map()
    if not m: return p
    pos=[(0 if pc.color==chess.WHITE else 6)+(pc.piece_type-1),r,c] for i,pc in m.items() for r,c in [divmod(i,8)]
    if pos: p[tuple(zip(*pos))]=1.0
    return p

def move_to_index(mv:chess.Move,b:chess.Board):
    f,t=mv.from_square,mv.to_square
    if b.is_castling(mv): return f*73+(0 if chess.square_file(t)>chess.square_file(f) else 1)
    pr=mv.promotion
    if pr:
        df,dr=chess.square_file(t)-chess.square_file(f),chess.square_rank(t)-chess.square_rank(f)
        dirs={ (0,1):0,(1,1):1,(-1,1):2,(0,2):3 }.get((df,dr))
        if dirs is None: return None
        pcs=[chess.QUEEN,chess.ROOK,chess.BISHOP,chess.KNIGHT]
        if pr not in pcs: return None
        return f*73+2+dirs*4+pcs.index(pr)
    df,dr=chess.square_file(t)-chess.square_file(f),chess.square_rank(t)-chess.square_rank(f)
    dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]
    plane=None; po=18
    for i,(dx,dy) in enumerate(dirs):
        if i<8:
            for dist in range(1,8):
                if df==dx*dist and dr==dy*dist: plane=po+i*7+(dist-1); break
            if plane is not None: break
        elif df==dx and dr==dy: plane=po+8*7+(i-8); break
    if plane is None: return None
    idx=f*73+(plane-po)
    return idx if 0<=idx<4672 else None

def estimate_game_count(pgn_path:str):
    try:
        f=open(pgn_path,"r",encoding="utf-8",errors="ignore")
        fin=tot=0; ih=se=sr=False
        for l in f:
            l=l.strip()
            if l.startswith("[Event "): ih,se,sr=True,True,False; continue
            if ih and l.startswith("[Result "): tok=l.split()[1].strip('"'); sr=tok in ("1-0","0-1","1/2-1/2","*"); continue
            if l=="": tot+=1 if se else 0; fin+=1 if sr else 0; ih,se,sr=False,False,False; continue
        f.close(); return fin if fin>0 else tot if tot>0 else None
    except: return None

def iter_pgn_positions(pgn_path:str,max_games:int=None,worker_id:int=None,num_workers:int=None):
    f=open(pgn_path,"r",encoding="utf-8"); gi=0
    while True:
        if max_games is not None and gi>=max_games: break
        g=chess.pgn.read_game(f); 
        if g is None: break
        r=g.headers.get("Result","*"); gr=1.0 if r=="1-0" else -1.0 if r=="0-1" else 0.0 if r=="1/2-1/2" else None
        if gr is None: gi+=1; continue
        if worker_id is not None and num_workers is not None and gi%num_workers!=worker_id: gi+=1; continue
        b=g.board(); n=g
        while n.variations:
            mn=n.variation(0); mv=mn.move; bt=encode_board(b); mi=move_to_index(mv,b)
            if mi is None: b.push(mv); n=mn; continue
            p=torch.zeros(4672); p[mi]=1.0
            stm=b.turn; v=0.0 if gr==0 else 1.0 if (gr==1 and stm) or (gr==-1 and not stm) else -1.0
            il=0 if mn.variations else 1
            yield bt,p,torch.tensor([v]),torch.tensor([il])
            b.push(mv); n=mn
        gi+=1
