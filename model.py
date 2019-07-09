
from lib import *

##################### MODEL DEFINITION #####################

class SiameseModel(nn.Module):
    
    def __init__(self):
        
        super(SiameseModel, self).__init__()
        
        # Convolution 1 [03,400,700 -> 16,77,137]
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=20, stride=5, padding=0)
        self.relu1 = nn.ReLU() 
        
        # Max pool 1 [16,77,137 -> 16,38,68]
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution 2 [16,38,68 -> 32,36,66]
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2 [32,36,66 -> 32,18,33]
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolution 3 [32,18,33 -> 64,16,31]
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        
        # Max pool 3 [64,16,31 -> 64,8,15]
        self.maxpool3 = nn.MaxPool2d(kernel_size=[2,3], stride=2)
        
        # Fully connected (readout) [64*8*15 -> 156]
        self.fc1 = nn.Linear(64 * 8 * 15, 156) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        
        # Convolution 3
        out = self.cnn3(out)
        out = self.relu3(out)
        
        # Max pool 3 
        out = self.maxpool3(out)
        
        # Resize [100,64,8,15 -> 100,64*8*15]
        out = out.view(out.size(0), -1)    
        
        # Linear function (readout)
        out = self.fc1(out)
        
        return out

