import os
import sys
import math

import numpy as np
import scipy
from scipy.linalg import circulant
from scipy.sparse import  kron, identity, csr_matrix
from scipy.stats import qmc
import matplotlib.pyplot as plt
import torch

from two_d_data_set import *
from two_d_model import  deeponet
from test_deeponet import domain
from utils import count_trainable_params, extract_path_from_dir, save_uniqe, grf, bilinear_upsample,upsample, generate_random_matrix, plot_cloud, generate_grf
from constants import Constants

                                                                      
names=[(0,15,0,15)]
test_names=[(0,15,0,15)]
# names=[]
# for k in range(1,4):
#     for l in range(1,4):
#         names.append( (0,int(15/k),0,int(15/l)) )


            
def generate_domains(i1,i2,j1,j2):
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    x_ref=d_ref.x
    y_ref=d_ref.y
    return domain(x_ref[i1:i2], y_ref[j1:j2])
    
    
def generate_f_g(shape, seedf):
    

        f=generate_random_matrix(shape,seed=seedf)
        
        f=(f-np.mean(f))/np.std(f)
        if seedf==2 or seedf==800:
            print("const")
            f=f*0+1

       
        return f+1
    
def generate_data(names,  save_path, number_samples,Seed):
    X=[]
    Y=[]
    MASKS=[]
    F=[]
    DOMAINS=[]
    for l,dom in enumerate(names):
        d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
        f_ref=np.zeros(d_ref.nx*d_ref.ny)
        d=generate_domains(dom[0],dom[1], dom[2],dom[3])
        mask = np.zeros((len(f_ref),len(f_ref)))
        mask[:, d.non_valid_indices] = float('-inf')  
        
        MASKS.append(torch.tensor(mask, dtype=torch.float32))
        DOMAINS.append(torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32))
        
        f_temp=[]
        
        GRF=generate_grf(d_ref.X, d_ref.Y, n_samples=number_samples,l=0.1, seed=Seed)
        # GRF=generate_grf(d.X, d.Y, n_samples=number_samples, seed=Seed)
        for i in range(number_samples):
            # f=GRF[i]
            f=GRF[i][d.valid_indices]
            
            # try:
            #     f=generate_f_g(d.nx*d.ny, Seed)
            # except:
            #     f=generate_f_g(d.nx*d.ny, i)
            # f_ref[d.valid_indices]=f
            f_ref[d.valid_indices]=f
            f_temp.append(torch.tensor(f_ref, dtype=torch.float32))
            
            A,G=d.solver(f.reshape((d.nx,d.ny)))
            # A,G=d.solver(0*upsample(f[0],int(n/2)).reshape((n,n)),[ga,gb,gc,gd])
            u=scipy.sparse.linalg.spsolve(A, G)
            for j in range(len(d.X)):
                X1=[
                    torch.tensor([d.X[j],d.Y[j]], dtype=torch.float32),
                    i,
                    l
                    ]
                Y1=torch.tensor(u[j], dtype=torch.cfloat)
                save_uniqe([X1,Y1],save_path+'data/')
                X.append(X1)
                Y.append(Y1)
                
        F.append(f_temp)   
    save_uniqe(MASKS ,save_path+'masks/')
    save_uniqe(DOMAINS ,save_path+'domains/')
    save_uniqe(F ,save_path+'functions/')
    return 1      

# 

if __name__=='__main__':
    pass
# if False: 400 is good for n=20
    generate_data(names, Constants.train_path, number_samples=300, Seed=1)
    generate_data(test_names,Constants.test_path,number_samples=1, Seed=800)

else:
    pass    
    # train_data=extract_path_from_dir(Constants.train_path)
    # s_train=[torch.load(f) for f in train_data]
    # X_train=[s[0] for s in s_train]
    # Y_train=[s[1] for s in s_train]
    # train_dataset = SonarDataset(X_train, Y_train)

    # train_size = int(0.8 * len(train_dataset))
    # val_size = len(train_dataset) - train_size

    # start=time.time()
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     train_dataset, [train_size, val_size]
    # )
 

    # train_dataloader = create_loader(train_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=False)
    # val_dataloader=create_loader(val_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=False)
    








# if __name__=='__main__':

# test_data=extract_path_from_dir(Constants.test_path)
# s_test=[torch.load(f) for f in test_data]
# X_test=[s[0] for s in s_test]
# Y_test=[s[1] for s in s_test]
# test_dataset = SonarDataset(X_test, Y_test)
# test_dataloader=create_loader(test_dataset, batch_size=Constants.batch_size, shuffle=False, drop_last=False)

# inp, out=next(iter(test_dataset))


# # model=deeponet_f2(2, 60) 
# n=30
# model=deeponet(dim=2,f_shape=n**2, domain_shape=2, p=80) 

# inp, out=next(iter(test_dataloader))
# model(inp)
# print(f" num of model parameters: {count_trainable_params(model)}")
# model([X[0].to(Constants.device),X[1].to(Constants.device)])

