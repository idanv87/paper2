for v in dir():
    exec('del '+ v)
    del v

import os
import sys
import math
import pyamg
import psutil
from matplotlib.ticker import ScalarFormatter

import time

from scipy.sparse import csr_array, tril
import scipy.sparse
import scipy.sparse.linalg
from scipy.stats import qmc
import scipy
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from memory_profiler import memory_usage, profile

import sys


from utils import upsample, fft, solve_gmres, track, find_sub_indices, GS_method
from constants import Constants
from utils import  grf, evaluate_model, generate_grf

from two_d_data_set import *
from packages.my_packages import Gauss_zeidel, interpolation_2D, gs_new, Gauss_zeidel2

from two_d_model import  deeponet
from test_deeponet import domain
from main import generate_f_g
# 2024.07.03.19.25.04best_model.pth single 
# 2024.07.03.23.10.51best_model.pth two domains
from df_polygon import generate_example, generate_rect, generate_rect2, generate_example_2, generate_obstacle, generate_obstacle2, generate_two_obstacles, generate_example3
# 2024.06.24.11.17.51best_model.pth  what wli asked 0,8,0,15 twice
# 2024.06.06.12.21.27best_model.pth several domains
# 2024.06.07.09.57.24best_model.pth single domain 

# 2024.06.05.12.50.00best_model.pth small domain  
# 2024.05.16.19.26.50best_model.pth with dom

class IterationCounter:
    def __init__(self):
        self.num_NN_iterations = 0
        self.num_GS_iterations = 0
        self.num_gmres_iterations = 0

class TimeCounter:
    def __init__(self):
        self.num_NN = 0
        self.num_GS = 0
        self.num_gmres = 0

model=deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.06.07.09.57.24best_model.pth')
model.load_state_dict(best_model['model_state_dict'])

# layer1_params = sum(p.numel() for p in model.model1.branch2.parameters() if p.requires_grad)


# best_model=torch.load(Constants.path+'runs/'+'2024.06.07.09.57.24best_model.pth')


   

def NN( F, t1,t2,mask, model,v1,v2):
    f=torch.tensor(F.reshape(1,F.shape[0]),dtype=torch.float32).unsqueeze(-1)

         

    with torch.no_grad():
        
        f1=model.model1.branch1(model.model1.attention1(
            f,f,f, mask[0].unsqueeze(0) ).squeeze(-1))
        f1=f1.repeat(t1.shape[0],1)
        f2=model.model2.branch1(model.model2.attention1(
            f,f,f, mask[0].unsqueeze(0) ).squeeze(-1))
        f2=f2.repeat(t2.shape[0],1)
        # start=time.time()
        pred2=model.forward2([t1,t2, f1,f2, mask, v1, v2])
        # print(time.time()-start)
        
    return torch.real(pred2).numpy()+1J*torch.imag(pred2).numpy()


def hints(A,b,x0, J, alpha,X,Y,X_ref,Y_ref,dom,mask, valid_indices, model, good_indices):
    iter_counter=IterationCounter()

    v1=model.model1.branch2(model.model1.attention2(
            dom.unsqueeze(0),dom.unsqueeze(0),dom.unsqueeze(0), mask.unsqueeze(0) )).squeeze(-1)
         
    v2=model.model2.branch2(model.model2.attention2(
            dom.unsqueeze(0),dom.unsqueeze(0),dom.unsqueeze(0), mask.unsqueeze(0) )).squeeze(-1)
    int_points=np.vstack([X,Y]).T
    y1=torch.tensor(int_points,dtype=torch.float32).reshape(int_points.shape)
    t1=model.model1.trunk1(y1)
    t2=model.model2.trunk1(y1)

    v1=v1.repeat(y1.shape[0],1)
    v2=v2.repeat(y1.shape[0],1)
    mask= mask.unsqueeze(0).repeat(y1.shape[0], 1, 1)

    A=csr_array(A)
    L=csr_array(tril(A,k=0))
    U=A-L   
  

    err=[]
    color=[]
    spec1=[]
    spec2=[]
    start=time.time()
    for k in range(6000):
      
        if (k+1)%J==0:
        
            iter_counter.num_NN_iterations+=1

            f_real=(-A@x0+b).real[good_indices]
            f_imag=(-A@x0+b).imag[good_indices]
            # 
            # func_real=interpolation_2D(X,Y,f_real)
            # func_imag=interpolation_2D(X,Y,f_imag)
            # 
            # f_real=np.array(func_real(X_ref,Y_ref))
            # f_imag=np.array(func_imag(X_ref,Y_ref))
            s_real=np.std(f_real)/alpha
            s_imag=np.std(f_imag)/alpha
            f_ref_real=np.zeros(Constants.n**2)
            f_ref_imag=np.zeros(Constants.n**2)
            
            f_ref_real[valid_indices]=(f_real)/s_real
            f_ref_imag[valid_indices]=(f_imag)/s_imag
            
            # start1=time.time()
            corr_real=(NN(f_ref_real,t1,t2, mask, model,v1,v2))*s_real
            iter_counter.num_NN_iterations+=1
            # print(time.time()-start1)
            

            corr_imag=(NN(f_ref_imag,t1,t2, mask, model,v1,v2))*s_imag
            iter_counter.num_NN_iterations+=1
            corr=corr_real+1J*corr_imag
            x0=x0+corr
  
            
        else:
            # start=time.time()  
            # x0=Gauss_zeidel2(U,L,b,x0)
            iter_counter.num_gmres_iterations+=1
            x0,_,iter,_=solve_gmres(A,b,x0,maxiter=30, tol=1e-10)

        if k %50 ==0:   
            pass 
            print(np.linalg.norm(A@x0-b)/np.linalg.norm(b))
            print(k)
        try:
            pass
            # spectrum, freqx, freqy=fft((abs((A@x0-b)/b)).reshape((8,15)),8,15)
            # spec1.append(abs(spectrum[4,7]))
            # spec2.append(abs(spectrum[0,0]))
        except:
            pass

        err.append(np.linalg.norm(A@x0-b)/np.linalg.norm(b))
        if err[-1]<1e-10 or err[-1]>100:

            time_counter=time.time()-start
            print(f' hints took: {time.time()-start} with {k} iteration and with error {err[-1]}')
            break
            
    # return err, color, J, alpha,k, spec1, spec2   

    return err, color, J, alpha,k, iter_counter, time_counter 


def exp3b(model, sigma=0.1,l=0.2,mean=0):
    # A,dom,mask, X, Y,X_ref,Y_ref, valid_indices=generate_rect2()
    # torch.save((A,dom,mask, X, Y,X_ref,Y_ref, valid_indices), Constants.outputs_path+'rect225.pt')
    # A,dom,mask, X, Y,X_ref,Y_ref, valid_indices=torch.load(Constants.outputs_path+'rect225.pt')
    # A,dom,mask, X, Y,X_ref,Y_ref, valid_indices=generate_example_2(N=113)
    A,dom,mask, X, Y,X_ref,Y_ref, valid_indices=generate_example3(N=225)
    # torch.save((A,dom,mask, X, Y,X_ref,Y_ref, valid_indices), Constants.outputs_path+'L225.pt')     
    # A,dom,mask, X, Y,X_ref,Y_ref, valid_indices=torch.load(Constants.outputs_path+'L225.pt')
    # print(A.shape)
    # A,dom,mask, X,Y,X_ref,Y_ref, valid_indices=generate_example()
    # A,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_obstacle2(225)
    # torch.save((A,dom,mask, X, Y,X_ref,Y_ref, valid_indices), Constants.outputs_path+'obs225.pt') 
    # A,dom,mask, X, Y,X_ref,Y_ref, valid_indices=torch.load(Constants.outputs_path+'obs225.pt')
    # A,dom,mask, X, Y,X_ref,Y_ref, valid_indices=generate_rect(N=113)
    print(A.shape)
    good_indices=find_sub_indices(X,Y,X_ref,Y_ref)
    # F=generate_grf(X,Y,n_samples=20,sigma=1, l=0.7 ,mean=10)
    f_ref=np.zeros(225)
   
    all_iter=[]
    all_time=[]
    for i in range(1):
        b=np.random.normal(1,0.5,A.shape[0])
        f_ref[valid_indices]=b[good_indices]
        x0=(b+1J*b)*0.001
        
        # x,err,iters,time_counter=solve_gmres(A,b,x0)
        # print(time_counter)
        # print(iters)
        # print(err)
        err, color, J, alpha, iters, iter_counter, time_counter=hints(A,b,x0,J=70, alpha=0.2,X=X,Y=Y,X_ref=X_ref,Y_ref=Y_ref,dom=dom,mask=mask, valid_indices=valid_indices, model=model, good_indices=good_indices)  
        
        all_iter.append(iters)
        all_time.append(time_counter)

    torch.save({'X':X, 'Y':Y,'all_iter':all_iter, 'all_time':all_time,'err':err}, Constants.outputs_path+'output14.pt')     
    
exp3b(model)    
data=torch.load(Constants.outputs_path+'output14.pt')
print(np.mean(data['all_iter']))    
# print(np.std(data['all_iter']))     
print(np.mean(data['all_time']))    




















# def exp4(sigma=0.4,l=0.4,mean=1):
#     A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
#     F=generate_grf(X,Y,20, sigma,l,mean)
#     all_k=[]
#     for i in range(20):
#         b=F[i]
        
#         f_ref[valid_indices]=b
#         x0=(b+1J*b)*0.001
#         err, color, J, alpha, k=hints(A,b,x0,J=2, alpha=1,X=X,Y=Y,X_ref=X,Y_ref=Y,dom=dom,mask=mask, valid_indices=valid_indices)  
#         all_k.append(k)
#     torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output15.pt')    

# def exp5(sigma=0.4,l=0.4,mean=1):
#     A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect2(8)
#     F=generate_grf(X,Y,20, sigma,l,mean)
#     all_k=[]
#     for i in range(20):
#         b=F[i]
#         x0=(b+1J*b)*0.001
#         f_ref[valid_indices]=b
#         e,k=solve_gmres(A,b,x0)
#         # err, color, J, alpha, k=hints(A,b,x0,J=2, alpha=1,X=X,Y=Y,X_ref=X,Y_ref=Y,dom=dom,mask=mask, valid_indices=valid_indices)  
#         all_k.append(k)
#     torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output16.pt') 

  

# def exp6(sigma=1,l=0.7,mean=1):
#     A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_example_2()
#     F=generate_grf(X,Y,20, sigma,l,mean)
#     all_k=[]
#     for i in range(20):
#         b=F[i]
#         func=interpolation_2D(X,Y,b)
#         f_ref[valid_indices]=func(X_ref,Y_ref)
#         x0=(b+1J*b)*0.001
        
#         err, color, J, alpha, k=hints(A,b,x0,J=5, alpha=1,X=X,Y=Y,X_ref=X_ref,Y_ref=Y_ref,dom=dom,mask=mask, valid_indices=valid_indices)  
#         all_k.append(k)
#     torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output17.pt')  
    
# def exp7(sigma=1,l=0.7,mean=1):
#     A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_obstacle2(29)
#     F=generate_grf(X,Y,n_samples=20,sigma=1, l=0.7 ,mean=1)
#     # F=generate_grf(X,Y,20, sigma,l,mean)
#     all_k=[]
#     for i in range(20):
#         b=F[i]
#         func=interpolation_2D(X,Y,b)
#         f_ref[valid_indices]=func(X_ref,Y_ref)
#         x0=(b+1J*b)*0.001
        
#         err, color, J, alpha, k=hints(A,b,x0,J=10, alpha=1,X=X,Y=Y,X_ref=X_ref,Y_ref=Y_ref,dom=dom,mask=mask, valid_indices=valid_indices)  
#         all_k.append(k)
#     # torch.save({'X':X, 'Y':Y,'J':J,'all_k':all_k}, Constants.outputs_path+'output18.pt')     

# def exp8(sigma=1,l=0.7,mean=1):
#     A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_obstacle2(29)
#     F=generate_grf(X,Y,20, sigma,l,mean)
#     all_k=[]
#     for i in range(20):
#         b=F[i]
#         func=interpolation_2D(X,Y,b)
#         f_ref[valid_indices]=func(X_ref,Y_ref)
#         x0=(b+1J*b)*0.001
#         e,it=solve_gmres(A,b,x0)
#         # err, color, J, alpha, k=hints(A,b,x0,J=10, alpha=1,X=X,Y=Y,X_ref=X_ref,Y_ref=Y_ref,dom=dom,mask=mask, valid_indices=valid_indices)  
#         all_k.append(it)
#         J='None'
#     torch.save({'X':X, 'Y':Y,'J':J,'all_k':all_k}, Constants.outputs_path+'output19.pt')  

   

# def exp9(sigma=1,l=0.7,mean=1):
#     A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_two_obstacles()
#     F=generate_grf(X,Y,20, sigma,l,mean)
#     all_k=[]
#     for i in range(20):
#         b=F[i]
#         func=interpolation_2D(X,Y,b)
#         f_ref[valid_indices]=func(X_ref,Y_ref)
#         x0=(b+1J*b)*0.001
        
#         err, color, J, alpha, k=hints(A,b,x0,J=10, alpha=1,X=X,Y=Y,X_ref=X_ref,Y_ref=Y_ref,dom=dom,mask=mask, valid_indices=valid_indices)  
#         all_k.append(k)
#     torch.save({'X':X, 'Y':Y,'J':J,'all_k':all_k}, Constants.outputs_path+'output20.pt')  

# def exp10(sigma=0.1,l=0.1,mean=0):
#     err=[]
#     alpha=1
#     A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect()
#     # A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
#     X_ref=X
#     Y_ref=Y
#     d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
#     # F=generate_grf(d_ref.X, d_ref.Y, n_samples=1, seed=1)[0]
#     # F=F[valid_indices]
#     F=generate_grf(X, Y, n_samples=50, seed=10)
#     for i in range(50):
#         sol=scipy.sparse.linalg.spsolve(A, F[i])
#         err.append(np.linalg.norm(single_hints(F[i], alpha,X,Y,X_ref,Y_ref,dom,mask, valid_indices)-sol)/np.linalg.norm(sol))
#         # print(err[-1])
#     print(np.mean(err))
#     print(np.std(err) )
#     # spectrum, freqx, freqy=fft(err.reshape((8,15)),8,15)
#     # print(abs(spectrum[0,1]))
#     # print(abs(spectrum[4,7]))
    
    
   
 
# def exp11(sigma=0.1,l=0.1,mean=0):
#     A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect()
#     # A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
#     F=generate_grf(X,Y,l=0.1,n_samples=1, seed=10)
#     all_k=[]
#     for i in range(1):
#         b=F[i]
#         f_ref[valid_indices]=b
#         x0=(b+1J*b)*0.001
#         err, color, J, alpha, k,spec1,spec2=hints(A,b,x0,J=2, alpha=1,X=X,Y=Y,X_ref=X,Y_ref=Y,dom=dom,mask=mask, valid_indices=valid_indices)  
#         torch.save({'low':spec1,'high':spec2}, Constants.outputs_path+'spec_mult.pt')  
#         all_k.append(k)
       
# # spec_single=torch.load(Constants.outputs_path+'spec_single.pt')        
# # spec_mult=torch.load(Constants.outputs_path+'spec_mult.pt')

# # plt.plot(spec_single['low'][:40])  
# # plt.plot(spec_mult['low'][:40],color='red')
# # plt.show()

# def exp12(sigma=0.1,l=0.1,mean=0):
#     # A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
#     A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect()
#     # F=generate_grf(X,Y,20)
#     F=generate_grf(X,Y,n_samples=20,sigma=0.4, l=0.4 ,mean=1)
    
#     all_k=[]
#     for i in range(20):
        
        
#         b=F[i]
#         u=scipy.sparse.linalg.spsolve(A, b)
#         f_ref[valid_indices]=b
#         x0=(b+1J*b)*0.001
#         e,it=solve_gmres(A,b,x0)
        
#         # err, color, J, alpha, k=hints(A,b,x0,J=5, alpha=1,X=X,Y=Y,X_ref=X,Y_ref=Y,dom=dom,mask=mask, valid_indices=valid_indices)  
#         all_k.append(it)
        
#     torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output20.pt') 

# # exp12()    
# # data=torch.load(Constants.outputs_path+'output20.pt')
# # print(np.mean(data['all_k']))    
# # print(np.std(data['all_k']))      

   

# def exp13(sigma=0.1,l=0.2,mean=0):
#     alpha=1
#     A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
#     # A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect()
#     F=generate_grf(X,Y,1,seed=2)
    
    
#     all_k=[]
#     for i in range(20):

#         u=F[i]*0
#         u[32:35]=F[i][32:35]
#         u[32:35]=F[i][47:50]
#         u[32:35]=F[i][62:65]
#         u[40:42]=F[i][40:42]
#         u[55:57]=F[i][55:57]
#         u[70:72]=F[i][70:72]
#         u[85:87]=F[i][85:87]
#         u[146:149]=F[i][146:149]
#         u[154:157]=F[i][154:157]
#         u=F[i]
#         b=A@(u/480)
        

#         sol=scipy.sparse.linalg.spsolve(A, b)
#         X_ref=X
#         Y_ref=Y
#         err=single_hints(b, alpha,X,Y,X_ref,Y_ref,dom,mask, valid_indices)-sol
#         # print(np.linalg.norm(err))
#         print(np.linalg.norm(err)/np.linalg.norm(sol))




# # def eval(f, alpha,X,Y,dom,mask, X_ref=None, Y_ref=None):
    
# #     f_real=f.real
# #     f_imag=f.imag
# #     # try:
# #     func_real=interpolation_2D(X,Y,f_real)
# #     func_imag=interpolation_2D(X,Y,f_imag)
# #     f_real=np.array(func_real(X_ref,Y_ref))
# #     f_imag=np.array(func_imag(X_ref,Y_ref))
# #     # except:
# #     #     pass    
    

# #     s_real=np.std(f_real)/alpha*0+1
# #     s_imag=np.std(f_imag)/alpha*0+1
# #     f_ref_real=np.zeros(Constants.n**2)
# #     f_ref_imag=np.zeros(Constants.n**2)
    
    
# #     f_ref_real[valid_indices]=(f_real)/s_real
# #     f_ref_imag[valid_indices]=(f_imag)/s_imag

# #     corr_real=(NN(f_ref_real,X,Y, dom, mask))*s_real
# #     corr_imag=(NN(f_ref_imag,X,Y, dom, mask))*s_imag
# #     # corr_real=(NN(f_ref_real,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_real)/s_real)*s_real
# #     # corr_imag=(NN(f_ref_imag,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_imag)/s_imag)*s_imag
# #     corr=corr_real+1J*corr_imag


# #     return corr   
   
# # A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
# # A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect2(8)
# # X_ref=None
# # Y_ref=None
# # A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_example_2()
# # l,v=gs_new(A.todense())
# # fig, ax1 = plt.subplots(1, 1, figsize=(4, 4)) 
# # V0=[]
# # V1=[]
# # V2=[]
# # # print(v.shape)
# # for j in [10,20,30,40,50,60]:
# #     V0.append(np.linalg.norm(eval(l[0]**j*v[:,0],0.5,X,Y,dom,mask,X_ref, Y_ref)/v[:,0]/l[0]**(j+1)))
# #     V1.append(np.linalg.norm(eval(l[1]**j*v[:,1],0.5,X,Y,dom,mask, X_ref,Y_ref)/v[:,1]/l[1]**(j+1)))
# #     # V2.append(np.linalg.norm(eval(l[2]**j*v[:,2],0.5,X,Y,dom,mask)/l[2]**j))

# # ax1.plot(V0, label='0')    
# # ax1.plot(V1, label='1')  
# # # # ax1.plot(V2, label='2')  
# # print(abs(l[0]))
# # print(abs(l[1]))
# # print(abs(l[2]))
# # # print(abs(l[-1]))
# # plt.legend()
# # plt.show()



# def single_hints(f, alpha,X,Y,X_ref,Y_ref,dom,mask, valid_indices):
#             f_real=f.real
#             f_imag=f.imag
#             s_real=np.std(f_real)
#             s_imag=np.std(f_imag)
#             f_ref_real=np.zeros(Constants.n**2)
#             f_ref_imag=np.zeros(Constants.n**2)
            
            
#             f_ref_real[valid_indices]=(f_real)
#             f_ref_imag[valid_indices]=(f_imag)
#             start=time.time()
#             corr_real=(NN(f_ref_real,X,Y, dom, mask, model))
#             print(time.time()-start)
#             corr_imag=(NN(f_ref_imag,X,Y, dom, mask, model))
#             # corr_real=(NN(f_ref_real,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_real)/s_real)*s_real
#             # corr_imag=(NN(f_ref_imag,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_imag)/s_imag)*s_imag
            
            
#             return corr_real+1J*corr_imag



# # def exp1(model):
# #     A,f_ref,f,dom,mask, X,Y, valid_indices=generate_example()
# #     # A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect()
# #     X_ref=X
# #     Y_ref=Y
# #     good_indices=find_sub_indices(X,Y,X_ref,Y_ref)
# #     F=generate_grf(X,Y,20)
# #     all_k=[]
# #     for i in range(20):
# #         b=F[i]
# #         f_ref[valid_indices]=b
# #         x0=(b+1J*b)*0.001
        
# #         def func():
# #             # err,k=solve_gmres(A,b,x0=x0)
# #             # print(k)
# #             err, color, J, alpha, k=hints(A,b,x0,J=50, alpha=0.2,X=X,Y=Y,X_ref=X_ref,Y_ref=Y_ref,dom=dom,mask=mask, valid_indices=valid_indices, model=model, good_indices=good_indices)  
# #             return  k
        
        
# #         all_k.append(func())
# #         torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output14.pt')     

# #         # err, color, J, alpha, k=hints(A,b,x0,J=5, alpha=1,X=X,Y=Y,X_ref=X,Y_ref=Y,dom=dom,mask=mask, valid_indices=valid_indices)  
# #         # all_k.append(k)
# #     # torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output11.pt')    
# # # exp1(model)
# # # data=torch.load(Constants.outputs_path+'output14.pt')
# # # print(np.mean(data['all_k']))    
# # # print(np.std(data['all_k'])) 


# # def exp2(model):
# #     A,f_ref,f,dom,mask, X,Y, valid_indices=generate_rect()
# #     X_ref=X
# #     Y_ref=Y
# #     good_indices=find_sub_indices(X,Y,X_ref,Y_ref)
# #     F=generate_grf(X,Y,n_samples=20,sigma=0.1, l=0.1 ,mean=1)
# #     all_k=[]
    
# #     for i in range(20):
# #         b=F[i]
# #         x0=(b+1J*b)*0.001
# #         f_ref[valid_indices]=b
# #         # e,k=solve_gmres(A,b,x0)
        
# #         err, color, J, alpha, k=hints(A,b,x0,J=2, alpha=0.2,X=X,Y=Y,X_ref=X,Y_ref=Y,dom=dom,mask=mask, valid_indices=valid_indices, model=model, good_indices=good_indices )  
# #         all_k.append(k)
# #     torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output13.pt') 

# # # exp2(model)    
# # # data=torch.load(Constants.outputs_path+'output13.pt')
# # # print(np.mean(data['all_k']))    
# # # print(np.std(data['all_k'])) 

# # def exp3(model, sigma=0.1,l=0.2,mean=0):
# #     A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_example_2()
# #     good_indices=find_sub_indices(X,Y,X_ref,Y_ref)
# #     F=generate_grf(X,Y,n_samples=20,sigma=1, l=0.7 ,mean=10)
    
# #     iter_number=[]
# #     for i in range(20):
# #         b=F[i]
# #         func=interpolation_2D(X,Y,b)
# #         f_ref[valid_indices]=func(X_ref,Y_ref)
# #         x0=(b+1J*b)*0.00001
        
# #         start=time.time()
# #         def func():
            
# #             # err,k=solve_gmres(A,b,x0)
# #             # print(err)
# #             # print(err)
# #             err, color, J, alpha, k=hints(A,b,x0,J=4, alpha=0.2,X=X,Y=Y,X_ref=X_ref,Y_ref=Y_ref,dom=dom,mask=mask, valid_indices=valid_indices, model=model, good_indices=good_indices)  
            
# #             return  k
        
        
# #         all_k.append(func())

# #     torch.save({'X':X, 'Y':Y,'all_k':all_k}, Constants.outputs_path+'output14.pt')     
    
# # # exp3(model)    
# # # data=torch.load(Constants.outputs_path+'output14.pt')
# # # print(np.mean(data['all_k']))    
# # # print(np.std(data['all_k'])) 
    