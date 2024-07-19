>kil
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import multiprocessing
import timeit
import datetime
import os
import time
import torch
from scipy import interpolate
from packages.my_packages import interpolation_2D, Restriction_matrix, Gauss_zeidel, gmres
from hints import deeponet
from utils import norms, calc_Robin, solve_subdomain, solve_subdomain2, grf
import random
from random import gauss
import scipy
from scipy.sparse import csr_matrix, kron, identity
from scipy.linalg import circulant
from scipy.sparse import block_diag
from scipy.sparse import vstack

# from jax.scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve, cg


import timeit


from constants import Constants
from scipy import signal
import matplotlib.pyplot as plt
def NN(F,u1,u2,R1,R2,x1,x2,y1,y2,X1,Y1):

    
    # u1=R1@u
    # u2=R2@u
    ROB1=calc_Robin(u2.real.reshape((len(x2)-1),len(y2)-2),x2[1]-x2[0],Constants.l,0)
    A1,G1=solve_subdomain2(x1,y1,R1@F,ROB1,l=Constants.l, side=1)
    res1=-A1@u1+G1
    p1=math.sqrt(np.var(res1.real))+1e-10
    q1=math.sqrt(0.02)
    factor1=q1/p1
    p2=math.sqrt(np.var(res1.imag))+1e-10
    q2=math.sqrt(0.02)
    factor2=q2/p2

    res_tilde=res1.real*factor1+1J*res1.imag*factor2
    xx=deeponet(res_tilde,X1.flatten(), Y1.flatten(),side=1)
    corr1=xx.real/factor1+1J*xx.imag/factor2
    u1=u1+corr1
    
    
    
    # res1=-A1@u1+G1
    # corr1=scipy.sparse.linalg.spsolve(A1, res1)
    # u1=u1+corr1
    
    ROB2=calc_Robin(u1.real.reshape((len(x1)-1),len(y1)-2),x1[1]-x1[0],Constants.l,1)
    A2,G2=solve_subdomain2(x2,y2,R2@F,ROB2,l=Constants.l, side=0)
    res2=-A2@u2+G2
    corr2=scipy.sparse.linalg.spsolve(A2, res2)
    u2=u2+corr2
    # print(np.linalg.norm(corr1))
    return u1, u2, corr1, corr2

def laplacian_matrix(x):
        dx=x[1]-x[0]
        Nx = len(x[1:-1])
        kernel = np.zeros((Nx, 1))
        kernel[-1] = 1.
        kernel[0] = -2.
        kernel[1] = 1.
        D2 = circulant(kernel)
        D2[0, -1] = 0.
        D2[-1, 0] = 0.
        return D2/dx/dx



n=21
x=np.linspace(0,1,n)
y=np.linspace(0,1,n)
L=csr_matrix(kron(laplacian_matrix(x), identity(len(y)-2)))+csr_matrix(kron(identity(len(x)-2), laplacian_matrix(y)))

ev,V=scipy.sparse.linalg.eigs(-L,k=15,return_eigenvectors=True,which="SR")
X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing='ij') 
# F=V[:,4]
# M=L+Constants.k*scipy.sparse.identity(L.shape[0])
M=L+Constants.k*scipy.sparse.identity(L.shape[0])
# solution=scipy.sparse.linalg.spsolve(M,F)
# u0=np.zeros(L.shape[0])
# for k in range(1):
#     u0=gmres(M.todense(),F,u0,1)[0]
#     print(np.linalg.norm(u0-solution)/np.linalg.norm(solution))
# F=grf(X.flatten(), 1,seed=2 )[0]
# F=f[0]
F=(-(2*math.pi**2-Constants.k)*np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()
solution=scipy.sparse.linalg.spsolve(M,F)
# g_truth=(np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()



x1=x[:int(0.5*(n-1))+1]
y1=y
x2=x[int(0.5*(n-1)):]
y2=y
X1, Y1 = np.meshgrid(x1[1:], y1[1:-1], indexing='ij') 
X2, Y2 = np.meshgrid(x2[:-1], y2[1:-1], indexing='ij') 
R1,R2=Restriction_matrix(X,Y,X1,Y1,X2,Y2)
u=(np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()*0

# u1=deeponet(R1@F, X1.flatten(), Y1.flatten(),side=1)*0
# u2=deeponet(R2@F, X2.flatten(), Y2.flatten(),side=0)*0


for k in range(0000):
    J=2
    if ((k+1)%J ==0):
    # if True:
        u1=R1@u
        u2=R2@u
        for j in range(1):  
            u1,u2,corr1,corr2=NN(F,u1,u2,R1,R2,x1,x2,y1,y2, X1,Y1)
            # print(np.linalg.norm(corr1))
            
      
        u1=u1.reshape((len(x1)-1,len(y1)-2)).copy()
        u2=u2.reshape((len(x2)-1,len(y2)-2)).copy()
        u=np.vstack((u1[:-1,:],u2)).flatten()
    else:
        u=Gauss_zeidel(M.todense(), F, u,theta=1)[0]  
        print(np.linalg.norm(u-(np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()))   

    # print(np.linalg.norm(u-(np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()))
    # print(np.linalg.norm((np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()-solution))
    # print(np.linalg.norm(u11-R1@solution))
#     # J=1
    # if  ((k+1) %J)==0:
    #     u=NN(F,u,R1,R2,x1,x2,y1,y2)
    # else:
    #     pass
    #     # u=Gauss_zeidel(M.todense(), F, u,theta=1)[0]
        
    # print(np.linalg.norm(u-solution))



# u=(np.sin(math.pi*X*3)*np.sin(math.pi*Y))
# u1=R1@(u.flatten())
# u2=R2@(u.flatten())

# u1=u1.reshape((10,19))
# u2=u2.reshape((10,19))
# u_tilde=np.vstack([u1,u2[1:,:]])
# print(u_tilde-u)

solution1=R1@solution
solution2=R2@solution

total_err=[]
experment_path=Constants.path+'runs/'
from two_d_model import Deeponet

# model_r=Deeponet(2,[10,19])
# model_c=Deeponet(2,[10,19])
# best_model=torch.load(experment_path+'2024.04.04.13.51.23best_model.pth')
# model_r.load_state_dict(best_model['model_state_dict']) 
# best_model=torch.load(experment_path+'2024.04.04.12.04.15best_model.pth')
# model_c.load_state_dict(best_model['model_state_dict']) 
u1=R1@u
u2=R2@u
ROB1=calc_Robin(u2.real.reshape((len(x2)-1),len(y2)-2),x2[1]-x2[0],Constants.l,0)
A1,G1=solve_subdomain2(x1,y1,R1@F,ROB1,l=Constants.l, side=1)
for k in range(2000):
    J1=3
    res1=-A1@u1+G1
    if ((k+1) %J1)==0:
        p1=math.sqrt(np.var(res1.real))+1e-10
        q1=math.sqrt(0.02)
        factor1=q1/p1
        p2=math.sqrt(np.var(res1.imag))+1e-10
        q2=math.sqrt(0.02)
        factor2=q2/p2
        start_time = time.time()
        res_tilde=res1.real*factor1+1J*res1.imag*factor2
        xx=deeponet(res_tilde, X1.flatten(), Y1.flatten(),side=1)
        corr1=xx.real/factor1+1J*xx.imag/factor2
        u1=u1+corr1
    else:    
        u1=Gauss_zeidel(A1.todense(), G1,u1,theta=1)[0]
    print(np.linalg.norm(A1@u1-G1))  


# A1,G1=solve_subdomain2(x1,y1,R1@F,ROB1,l=Constants.l, side=1)
# for k in range(2000):
#     J1=3
#     J2=100
#     res1=-A1@u1+G1
#     if ((k+1) %J1)==0:
#         p1=math.sqrt(np.var(res1.real))+1e-10
#         q1=math.sqrt(0.02)
#         factor1=q1/p1
#         p2=math.sqrt(np.var(res1.imag))+1e-10
#         q2=math.sqrt(0.02)
#         factor2=q2/p2
#         start_time = time.time()
#         res_tilde=res1.real*factor1+1J*res1.imag*factor2
#         xx=deeponet(res_tilde, X1.flatten(), Y1.flatten(),side=1)
#         corr1=xx.real/factor1+1J*xx.imag/factor2
#         u1=u1+corr1
#     else:    
#         u1=Gauss_zeidel(A1.todense(), G1,u1,theta=1)[0]
#     print(np.linalg.norm(A1@u1-G1))    
 
 
 
 
 
 
 
# u1=R1@u    
# u2=R2@u     
for k in range(0000):
    J1=3000
    J2=10000
    ROB1=calc_Robin(u2.real.reshape((len(x2)-1),len(y2)-2),x2[1]-x2[0],Constants.l,0)
    A1,G1=solve_subdomain2(x1,y1,R1@F,ROB1,l=Constants.l, side=1)
    res1=-A1@u1+G1
    if ((k+1) %J1)==0:
    # if False:
        # start_time = time.time()
        p1=math.sqrt(np.var(res1.real))+1e-10
        q1=math.sqrt(0.02)
        factor1=q1/p1
        p2=math.sqrt(np.var(res1.imag))+1e-10
        q2=math.sqrt(0.02)
        factor2=q2/p2
        start_time = time.time()
        res_tilde=res1.real*factor1+1J*res1.imag*factor2
        xx=deeponet(res_tilde, X1.flatten(), Y1.flatten(),side=1)
        corr1=xx.real/factor1+1J*xx.imag/factor2
        u1=u1+corr1
        
        
    else:    
        # u1=Gauss_zeidel(A1.todense(),G1,u1,theta=1, iter=3)[0]

        corr1=scipy.sparse.linalg.spsolve(A1, res1)
        u1=u1+corr1

        
    
    err1=np.linalg.norm(u1-R1@solution)/np.linalg.norm(R1@solution)

    ROB2=calc_Robin(u1.real.reshape((len(x1)-1),len(y1)-2),x1[1]-x1[0],Constants.l,1)
    A2,G2=solve_subdomain2(x2,y2,R2@F,ROB2,l=Constants.l, side=0)
    res2=-A2@u2+G2
    if ((k+1) %J2)==0:
    # if False:    

        p1=math.sqrt(np.var(res1.real))+1e-10
        q1=math.sqrt(3.698418078955824)
        factor1=q1/p1
        p2=math.sqrt(np.var(res1.imag))+1e-10
        q2=math.sqrt(363.62700517944705)
        factor2=q2/p2
        res_tilde=res2.real*factor1+1J*res2.imag*factor2
        xx=deeponet(res_tilde, X2.flatten(), Y2.flatten(),side=0)
        corr2=xx.real/factor1+1J*xx.imag/factor2
        u2=u2+corr2
        # yy=scipy.sparse.linalg.spsolve(A1, res_tilde).real
        # print(np.linalg.norm(xx-yy)/np.linalg.norm(yy))
        
        
        
    else:    
        
        corr2=scipy.sparse.linalg.spsolve(A2, res2)
        u2=u2+corr2
        
    
   
    err2=np.linalg.norm(u2-R2@solution)/np.linalg.norm(R2@solution)
    # total_err.append((np.linalg.norm(corr1)))
    print(np.linalg.norm(err2+err1))
    # print((err1+err2)/2)
# torch.save(total_err,  '/Users/idanversano/Documents/project_geo_deeponet/tex/figures/err_k_100_hybrid.pt')
# torch.save(total_err,  '/Users/idanversano/Documents/project_geo_deeponet/tex/figures/err_k_100_schwartz.pt')

fig = plt.figure()
ax = plt.axes()
A=torch.load ('/Users/idanversano/Documents/project_geo_deeponet/tex/figures/err_k_100_hybrid.pt')
B=torch.load ('/Users/idanversano/Documents/project_geo_deeponet/tex/figures/err_k_100_schwartz.pt')
ax.plot(np.log(np.array(range(len(A)))[1:]),np.log(A[1:]), c='r',label='hybrid')
ax.plot(np.log(np.array(range(len(B)))[1:]),np.log(B[1:]), c='b',label='schwartz')

ax.set_xlabel('log(step)')
ax.set_ylabel('log(err)')
ax.legend()
# plt.show()




if False:
    u2=(np.sin(math.pi*X2)*np.sin(math.pi*Y2))*0
    # u1=R1@u
    # u2=R2@u
    for k in range(1000):
        
        ROB1=calc_Robin(u2.real,x2[1]-x2[0],Constants.l,0)
        # print(ROB1)
        # f=interpolation_2D(X.flatten(),Y.flatten(),F )
        # g=interpolate.InterpolatedUnivariateSpline(y2[1:-1], ROB1)
        # u=deeponet(f,g,X.flatten(),Y.flatten()).reshape((len(x1)-1),len(y1)-2)
        u1=(solve_subdomain(x1,y1,R1@F,ROB1,l=Constants.l, side=1)).reshape((len(x1)-1),len(y1)-2)

        g_truth=(np.sin(math.pi*X1)*np.sin(math.pi*Y1)).flatten()
        # print(np.linalg.norm(u-u1)/np.linalg.norm(u1))

        print(np.linalg.norm((solution1.reshape((len(x1)-1),len(y1)-2))[1:-1,1:-1]-u1[1:-1,1:-1]))
        ROB2=calc_Robin(u1.real,x1[1]-x1[0],Constants.l,1)
        u2=(solve_subdomain(x2,y2,R2@F,ROB2,l=Constants.l, side=0)).reshape((len(x2)-1),len(y2)-2)
        # print(np.linalg.norm(np.sin(math.pi*X)*np.sin(math.pi*Y)-u2)/np.linalg.norm(u2))
        

   
# def dd_sover(n):
#     x=np.linspace(0,1,n)
#     y=np.linspace(0,1,n)
#     L=csr_matrix(kron(laplacian_matrix(x), identity(len(y)-2)))+csr_matrix(kron(identity(len(x)-2), laplacian_matrix(y)))

#     X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing='ij') 

#     M=L+Constants.k*scipy.sparse.identity(L.shape[0])
#     F=(-(2*math.pi**2-Constants.k)*np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()
#     # solution=scipy.sparse.linalg.spsolve(M,F)
#     g_truth=(np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()
#     # return (x[1]-x[0])*np.linalg.norm(scipy.sparse.linalg.spsolve(M, F)-g_truth.flatten())
#     x1=x[:int(0.5*(n-1))+1]
#     y1=y
#     x2=x[int(0.5*(n-1)):]
#     y2=y
#     X1, Y1 = np.meshgrid(x1[1:], y1[1:-1], indexing='ij') 
#     X2, Y2 = np.meshgrid(x2[:-1], y2[1:-1], indexing='ij') 


#     # print(np.cos(0.5)*np.sin(y1[1:-1]))

#     D=np.array(range(len(X.flatten())))
#     D1=np.array(range(len(X1.flatten())))
#     D2=np.array(range(len(X1.flatten())-len(y[1:-1]),len(X.flatten())))

#     R1=Restriction(D,D1)
#     R2=Restriction(D,D2)
#     u=(np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()
#     u1=R1@u
#     u2=R2@u
#     corr1=10
#     corr2=10
#     tol=1e-13
#     max_iter=5000
#     count=0
#     while (count<max_iter) and (np.linalg.norm(corr1+corr2)>tol):
#         count+=1
#         u1,u2, corr1, corr2=NN(F,u1,u2,R1,R2,x1,x2,y1,y2)
#     u1=u1.reshape((len(x1)-1,len(y1)-2)).copy()
#     u2=u2.reshape((len(x2)-1,len(y2)-2)).copy()
#     u=np.vstack((u1[:-1,:],u2)).flatten()
#     print(np.linalg.norm(corr1))
#     # return (1/len(u))*np.linalg.norm(u-g_truth.flatten())
#     return np.max(abs(u-g_truth.flatten()))

# all_n=[21, 31]
# all_h=[(1/(n-1)) for n in all_n]
# err=[]
# for n in all_n:
#     err.append(dd_sover(n))

# torch.save(err,  '/Users/idanversano/Documents/project_geo_deeponet/tex/figures/err.pt')
# Err=torch.load ('/Users/idanversano/Documents/project_geo_deeponet/tex/figures/err.pt')
# all_h.reverse()
# Err.reverse()
# x=[]
# y=[]
# for i in range(len(all_h)):
#     try:
#         x.append(np.log(all_h[-1-i])-np.log(all_h[-1-i-1]))
#         y.append(np.log(Err[-1-i])-np.log(Err[-1-i-1]))
#     except:
#         pass
# print([y[i]/x[i] for i in range(len(x))])

# # plt.scatter(np.log(all_h),np.log(Err))
# # plt.show()

    
