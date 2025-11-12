import torch; from model import ChessNet
def cmp(m1,m2,bs=4,b=3):
    d="cuda" if torch.cuda.is_available() else "cpu"; m1.to(d).eval(); m2.to(d).eval(); m2.load_state_dict(m1.state_dict())
    for i in range(b):
        x=torch.randn(bs,18,8,8,device=d)
        with torch.no_grad(): p1,v1=m1(x); p2,v2=m2(x)
        pc,vc=torch.allclose(p1,p2,1e-5),torch.allclose(v1,v2,1e-5)
        print(f"B{i+1}: Policy {'PASS' if pc else 'FAIL'} max={(p1-p2).abs().max():.2e}, Value {'PASS' if vc else 'FAIL'} max={(v1-v2).abs().max():.2e}")
if __name__=="__main__": cmp(ChessNet(),ChessNet())
