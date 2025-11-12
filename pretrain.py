import os, torch, gc, argparse
from torch.utils.data import IterableDataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
from model import load_model
from utils.pgn_dataset import iter_pgn_positions, estimate_game_count

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu"
DATA_PATH = os.path.join("data","lichess_elite_2023-01.pgn")
USE_TORCH_COMPILE, USE_FP8 = False, False

if DEVICE=="cuda":
    try: torch.backends.cuda.matmul.fp32_precision="tf32"
    except: pass
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False

class PGNDataset(IterableDataset):
    def __init__(self,pgn_path,max_games=None): self.pgn_path,self.max_games=pgn_path,max_games
    def __iter__(self):
        try: from torch.utils.data import get_worker_info; w=get_worker_info()
        except: w=None
        return iter_pgn_positions(self.pgn_path,max_games=self.max_games,worker_id=getattr(w,"id",None),num_workers=getattr(w,"num_workers",None))

def train_batch(model,opt,X,P,V,wP,wV):
    model.train()
    X,P,V=[t.to(DEVICE,non_blocking=True) for t in (X,P,V)]
    opt.zero_grad(set_to_none=True)
    ctx=torch.amp.autocast("cuda" if DEVICE=="cuda" else "mps",dtype=torch.bfloat16) if DEVICE!="cpu" else nullcontext()
    with ctx:
        logP,v=model(X)
        loss=wP*(-(P*logP).sum(1).mean())+wV*((v-V)**2).mean()
    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    val=float(loss.item()); del loss,X,P,V,logP,v; return val

from contextlib import nullcontext

def pretrain(pgn_path=DATA_PATH,model_path_out="chess_model.pt",init_model_path=None,max_games=None,epochs=1,batch_size=512,lr=1e-3,wP=1.0,wV=1.0):
    assert os.path.exists(pgn_path),f"PGN not found: {pgn_path}"
    model=load_model(init_model_path,DEVICE)
    print(f"Params total={sum(p.numel() for p in model.parameters()):,} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if USE_TORCH_COMPILE: model=torch.compile(model)
    try: opt=optim.AdamW(model.parameters(),lr=lr,fused=DEVICE=="cuda")
    except: opt=optim.AdamW(model.parameters(),lr=lr)
    est=estimate_game_count(pgn_path)
    ds=PGNDataset(pgn_path,max_games)
    dl=DataLoader(ds,batch_size=batch_size,num_workers=8 if DEVICE=="cuda" else 0,pin_memory=DEVICE=="cuda",prefetch_factor=2,persistent_workers=DEVICE=="cuda")
    for e in range(epochs):
        print(f"\nEpoch {e+1}/{epochs}")
        total_games=None if est is None else min(est,max_games) if max_games else est
        pbar=tqdm(total=total_games,unit="games",desc=f"Pretraining games (epoch {e+1}/{epochs})")
        for B,P,V,L in dl:
            loss=train_batch(model,opt,B,P,V,wP,wV)
            try: pbar.update(int(L.sum().item()))
            except: pbar.update(sum(int(x) for x in L))
            pbar.set_postfix({"loss":f"{loss:.4f}"})
        pbar.close()
        torch.save(model.state_dict(),model_path_out); print(f"Saved model to {model_path_out}")
    gc.collect(); DEVICE=="cuda" and torch.cuda.empty_cache(); print("Pretraining complete.")

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--pgn",default=DATA_PATH); p.add_argument("--init"); p.add_argument("--out",default="chess_model.pt"); p.add_argument("--max-games",type=int); p.add_argument("--epochs",type=int,default=1); p.add_argument("--batch-size",type=int,default=512); p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--weight-policy",type=float,default=1.0); p.add_argument("--weight-value",type=float,default=1.0); return p.parse_args()

def main(): a=parse_args(); pretrain(a.pgn,a.out,a.init,a.max_games,a.epochs,a.batch_size,a.lr,a.weight_policy,a.weight_value)

if __name__=="__main__": main()
