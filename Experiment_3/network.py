# import libiary
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchinfo import summary

# Define network
class Net(nn.Module):
    # define each layers
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1),                              
                nn.ReLU(),                      
                nn.MaxPool2d(2,2),    
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),     
                nn.ReLU(),                      
                nn.MaxPool2d(2,2),                
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),     
                nn.ReLU(),                      
                nn.MaxPool2d(2,2),                
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),     
                nn.ReLU(),                      
                nn.MaxPool2d(2,2),                
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),     
                nn.ReLU(),                      
                nn.MaxPool2d(2,2),                
            )   
        self.layer6 = nn.Sequential(
            nn.Conv2d(16, 4, 3, 1, 1),     
                nn.ReLU(),                      
                nn.MaxPool2d(2,2),                
            )
        self.fc1 = nn.Linear(4 * 4 * 4, 8)
        self.fc2 = nn.Linear(8,1)

    # Define forward path
    def forward(self, x):

        x = self.layer1(x)
         
        x = self.layer2(x)
         
        x = self.layer3(x)
         
        x = self.layer4(x)
         
        x = self.layer5(x)
         
        x = self.layer6(x)
         
        x = x.view(x.size(0), -1)
         
        x = F.relu(self.fc1(x))
         
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        x = x.reshape(-1)
        return x


# Define FC network
class FCnet(nn.Module):
    # Define two layers
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32,8)
        self.fc2 = nn.Linear(8,1)
        # self.drop = nn.Dropout(0.6)

    # Define forward path
    def forward(self,x):
        # x = self.drop(x)
        x = F.relu(self.fc(x))
        # x = self.drop(x)
        x = self.fc2(x)

        return x

# network = Net()
# print(summary(network))
