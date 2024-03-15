
import numpy as np
import torch




dir = "/Users/mac/vs code projects/GUI/Communication/B.npy"
sp = np.load(dir)[1]
sp = torch.tensor(sp, dtype=torch.float32)[:2048]
sp = sp.reshape(-1 ,1 ,2048)
print(sp)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sp = sp.to(device)
net = torch.load("/Users/mac/vs code projects/GUI/Communication/demo/saved_models/2-CNN_2conv_2fc-361-0.972.pth", map_location=torch.device('cpu'))
out = net(sp)
print(out)