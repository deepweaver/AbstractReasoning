import torch
import torch.nn.functional as F
import numpy as np 
from torch import nn
from PIL import Image 
class CRModel(nn.Module):
    def __init__(self):
        super(CRModel, self).__init__()
        self.NUM_PANELS = 9
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cnn = nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv2d(1, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ).to(self.device)
    def panel_embeddings(self, panels):
        batch_size = panels.shape[0] # which is 1 or 128 ...
        panel_num = panels.shape[1] # which is 9 
        # 128, 9, 80, 80 
        embeddings = []
        # panel_embeddings = torch.zeros(()) 
        # (128, 32, 4, 4)
        # 4 total inputs for RNN, (input_size=32*4*4, hidden_size=64, num_layers=1)
        # 6 embeddings per RNN input seq
        inputs = []
        # dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.float
        for i in range(5):
            for j in range(4):
                m = torch.zeros((6, batch_size, 32*4*4)).to(self.device)
                cnnout = self.cnn(panels[:,i:i+1,:,:])  # (128, 32, 4, 4)
                m[i,:,:] = cnnout.view(batch_size, -1) 
                inputs.append(m) 
        self.rnn = nn.RNN(32*4*4, 64).to(self.device) 
        self.outputs = torch.zeros((batch_size,4)).to(self.device)
        self.linear = nn.Linear(64,1).to(self.device) 
        for i in range(5,9):
            cnnout = self.cnn(panels[:,i:i+1,:,:])
            inputs[i-5][5,:,:] = cnnout.view(batch_size, -1)
            _, h_n = self.rnn(inputs[i-5])
            self.outputs[:,i-5] = self.linear(h_n)[0,:,0]
            # outputs.append(self.linear(h_n))
            # four such passes produce four scalar scores
            # outputs.append()
        

        # a tuple with two elements
        # with size (6, 128, 64), (1, 128, 64)
        # print(outputs[0].shape)
        self.softmax = nn.Softmax(dim=1).to(self.device) 
        return self.softmax(self.outputs) 

        
        # self.linear()
    def forward(self, x):
        return self.panel_embeddings(x) 






if __name__=='__main__':
    batch_size = 128
    dataset_1b = np.zeros((batch_size,9,80,80), dtype=np.float64) 
    for i in range(batch_size):
        data = np.load('/Users/tiany/Downloads/novel.domain.transfer/analogy_novel.domain.transfer_train_normal_{}.npz'.format(600000-i))
        panels = data['image'].reshape((9,160,160)).astype(np.uint8)
        for j in range(9):
            # print(panels[j].shape)
            img = Image.fromarray(panels[j]).resize((80,80)) 
            dataset_1b[i,j,:,:] = np.asarray(img, dtype=np.float64)  

    dataset_1b_tch = torch.from_numpy(dataset_1b).float()
    # print(type(dataset_1b_tch))
    mdl = CRModel()
    result = mdl.forward(dataset_1b_tch) 
    print(result.shape)







