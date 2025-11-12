import torch, torch.nn as nn, torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,c): super().__init__(); self.conv1=nn.Conv2d(c,c,3,1,1); self.bn1=nn.BatchNorm2d(c); self.conv2=nn.Conv2d(c,c,3,1,1); self.bn2=nn.BatchNorm2d(c)
    def forward(self,x): r=x; x=F.relu(self.bn1(self.conv1(x))); x=self.bn2(self.conv2(x)); return F.relu(x+r)

class ChessNet(nn.Module):
    def __init__(self,num_res_blocks=6,channels=768):
        super().__init__()
        self.input_conv=nn.Conv2d(18,channels,3,1,1); self.input_bn=nn.BatchNorm2d(channels)
        self.res_blocks=nn.ModuleList([ResidualBlock(channels) for _ in range(num_res_blocks)])
        self.policy_conv=nn.Conv2d(channels,32,1); self.policy_bn=nn.BatchNorm2d(32); self.policy_fc=nn.Linear(32*8*8,4672)
        self.value_conv=nn.Conv2d(channels,32,1); self.value_bn=nn.BatchNorm2d(32); self.value_fc1=nn.Linear(32*8*8,256); self.value_fc2=nn.Linear(256,1)

    def forward(self,x):
        x=F.relu(self.input_bn(self.input_conv(x)))
        for b in self.res_blocks: x=b(x)
        p=F.log_softmax(self.policy_fc(F.relu(self.policy_bn(self.policy_conv(x))).view(x.size(0),-1)),dim=1)
        v=torch.tanh(self.value_fc2(F.relu(self.value_fc1(F.relu(self.value_bn(self.value_conv(x))).view(x.size(0),-1)))))
        return p,v

def load_model(path=None,device="cpu"):
    m=ChessNet()
    path and m.load_state_dict(torch.load(path,map_location=device))
    return m.to(device)
