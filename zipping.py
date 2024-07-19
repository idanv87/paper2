import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
class Tnet(nn.Module):
   def __init__(self, k=1):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input):
      # input.shape == (bs,n,3)
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix


class Transform(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=1)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(1,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
   def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.transform = Transform()
        
    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        return xb

if __name__=='__main__':
    model = PointNet()
    device='cpu'
    model.to(device)
    model.eval()
    print(model(torch.rand(32,1,4).to(device)).shape)
    # input_dim=10
    # input_vector1 = torch.rand(3,input_dim)

    # # Permute input_vector2 to make it a permutation of input_vector1
    # perm_indices1 = torch.randperm(input_dim)
    # input_vector1_permuted = input_vector1[:,perm_indices1]
    # print(input_vector1)
    # print(input_vector1_permuted)
    # Convert input vectors to tensors and add batch dimension

    # x1=input_vector1.unsqueeze(0)
    # x2=input_vector1_permuted.unsqueeze(0)
    # print(model(x1.to(device))-model(x2.to(device)))
    # print(model(x1.to(device)))
    # print(model(x1.to(device)).shape)

    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(x1.shape)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    # criterion = torch.nn.NLLLoss()
    # bs=outputs.size(0)
    # id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    # id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    # if outputs.is_cuda:
    #     id3x3=id3x3.cuda()
    #     id64x64=id64x64.cuda()
    # diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    # diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    # return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)