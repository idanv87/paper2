import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        
    def scaled_dot_product_attention(self, Q, K, V, mask):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        attn_probs = torch.softmax(attn_scores+mask, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

        
    def forward(self, Q, K, V, mask=None):
        Q =self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        return attn_output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.fc2(self.tanh(self.fc1(x)))
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
       
        
    def forward(self, x, mask):

        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x

class EncoderLayer2(nn.Module):
    def __init__(self, d_model, d_ff):
        super(EncoderLayer2, self).__init__()
        self.self_attn = MultiHeadAttention(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
       
        
    def forward(self, x,z):

        attn_output = self.self_attn(x, x, z, mask=None)
        x = self.norm1(z + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x

# x=torch.randn(1,400,1)
# y=torch.randn(1,400,2)
# e=EncoderLayer(1, 80)    
# d=EncoderLayer(2,80)
# c=EncoderLayer2(1,  80)  

# n=nn.Linear(2,1, bias=False)
# v=e(x,None)
# v.shape
# w=n(d(y,None))
# c(v,w).shape

