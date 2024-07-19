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
from packages.my_packages import interpolation_2D, Restriction_matrix, Gauss_zeidel, gmres2, Dx_backward, Dx_forward
# from hints import deeponet
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




class domain:
    def __init__(self,x,y):

        self.x=x 
        self.dx=x[1]-x[0]
        self.y=y
        self.nx=len(x)
        self.ny=len(y)
        self.dy=y[1]-y[0]
        xx,yy=np.meshgrid(self.x,self.y,indexing='ij')
        self.X=xx.flatten()
        self.Y=yy.flatten()
        self.D=csr_matrix(kron(self.calc_D_x(), identity(len(self.y))),dtype=np.cfloat)+csr_matrix(kron(identity(len(self.x)), self.calc_D_y()), dtype=np.cfloat)
        self.valid_indices, self.non_valid_indices=self.masking_coordinates()
       
        
    def  calc_D_x(self):   
        Nx = len(self.x)
        kernel = np.zeros((Nx, 1))
        kernel[-1] = 1.
        kernel[0] = -2.
        kernel[1] = 1.
        D2 = circulant(kernel).astype(complex)
        D2[0, -1] = 0.
        D2[-1, 0] = 0.
        D2[-1,-1]=-2-2*self.dx*Constants.l
        D2[-1,-2]=2   
        D2[0,0]=-2-2*self.dx*Constants.l
        D2[0,1]=2    
        return csr_matrix(D2/self.dx/self.dx   )
    
    def  calc_D_y(self):   
        Ny = len(self.y)
        kernel = np.zeros((Ny, 1))
        kernel[-1] = 1.
        kernel[0] = -2.
        kernel[1] = 1.
        D2 = circulant(kernel).astype(complex)
        D2[0, -1] = 0.
        D2[-1, 0] = 0.
        
        D2[-1,-1]=-2-2*self.dy*Constants.l
        D2[-1,-2]=2   
        D2[0,0]=-2-2*self.dy*Constants.l
        D2[0,1]=2    
        return csr_matrix(D2/self.dy/self.dy   )
    
        
    def solver(self,f=0,g=0):
        
        return self.D+Constants.k*scipy.sparse.identity(self.D.shape[0]), f
    
    def masking_coordinates(self):

        xx,yy=np.meshgrid(np.linspace(0,1, Constants.n),np.linspace(0,1, Constants.n),indexing='ij')
        X0=xx.flatten()
        Y0=yy.flatten()
        original_points=[(X0[i],Y0[i]) for i in range(len(X0))]
        points=np.array([(self.X[i],self.Y[i]) for i in range(len(self.X))])
        valid_indices=[]
        masked_indices=[]
        for j,p in enumerate(original_points):
           
            dist=[np.linalg.norm(np.array(p)-points[i]) for i in range(points.shape[0])]
            if np.min(dist)<1e-14:
                valid_indices.append(j)
            else:
                masked_indices.append(j)    
        return valid_indices, masked_indices    
            
# n=30    
# x=np.linspace(0,1,n)
# y=np.linspace(0,1,n)    
# d=domain(x,y,1,1)

    

def dd_block(n,u_global,s0,s1,f):
    u1=u_global[1:int((n-1)/2)]
    u0=u_global[int((n-1)/2)-1:-1]
    
    g=Dx_forward(u0.real,s0.dx)+Constants.l*u0[0]
    A1,G1=s1.solver(f[1:int((n-1)/2)],g)
    res1=-A1@u1+G1
    corr1=scipy.sparse.linalg.spsolve(A1,res1).real
    u1=u1+corr1
    
    g=(-Dx_backward(u1.real,s1.dx)+Constants.l*u1[-1])
    A0,G0=s0.solver(f[int((n-1)/2)-1:-1],g)
    res0=-A0@u0+G0
    corr0=scipy.sparse.linalg.spsolve(A0,res0).real
    u0=u0+corr0
    return u0,u1,np.concatenate([[0],u1[:-1],u0,[0]])

# from utils import upsample
# n=30
# x=np.linspace(0,1,n)
# y=np.linspace(0,1,n)
# d=domain(x,y,1,1)
# xx,yy=np.meshgrid(x,y,indexing='ij')
# f=grf(int((n/2))**2, 1,seed=1 )
# f=upsample(f[0],int(n/2))
# print(f.shape)
# A,b=d.solver(upsample(f[0],int(n/2)).reshape((n,n)),0)
# x0=b*0