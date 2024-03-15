import torch
from config import Config
import numpy as np
class CNN_2conv_2fc(torch.nn.Module):
    def __init__(self):
        super(CNN_2conv_2fc, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=16, padding=36),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2)
            )
        self.fc = torch.nn.Sequential(torch.nn.Linear(1024, 32),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.5),
                                      torch.nn.Linear(32, 4))

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        y_hat = torch.nn.functional.softmax(x, dim=1)
        return y_hat
    

class CNN_4conv_2fc(torch.nn.Module):
    def __init__(self):
        super(CNN_4conv_2fc, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=64, stride=16, padding=36),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2),
            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2),
            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2),
            )
        self.fc = torch.nn.Sequential(torch.nn.Linear(Config.fc_input_size, 32),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.5),
                                      torch.nn.Linear(32, 4))

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        y_hat = torch.nn.functional.softmax(x, dim=1)
        return y_hat


if __name__ == '__main__':
    dir = 'python_projects/CWRU-project/test0_4layercnn_10cls/CWRU_dataset_10_traintest/test/0/97_0.npy'
    sp = torch.tensor(np.load(dir), dtype=torch.float32)[:2048]
    sp = sp.reshape(-1,1,2048)
    net = CNN_4conv_2fc()
    print(net.__class__.__name__)
    out = net(sp)
    print(out)
