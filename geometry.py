import datetime
import os
import sys
import time

from shapely.geometry import Polygon as Pol2
from pylab import figure
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dmsh
import meshio
import optimesh
from pathlib import Path
import torch
from sklearn.cluster import KMeans
from scipy.interpolate import Rbf
from scipy.spatial.distance import euclidean, cityblock
import cmath

from utils import *
from constants import Constants
# from coords import Map_circle_to_polygon
from pydec.dec import simplicial_complex
from functions.functions import Test_function

from packages.my_packages import *
from scipy.sparse import csr_matrix, kron, identity
from scipy.linalg import circulant


class Rect:
    def __init__(self, x,y,side):
        self.x=x
        self.y=y
        self.side=side
        self.dx=x[1]-x[0]
        self.dy=y[1]-y[0]
        if self.side:
            self.X, self.Y = np.meshgrid(x[1:], y[1:-1], indexing='ij') 
        else:    
            self.X, self.Y = np.meshgrid(x[:-1], y[1:-1], indexing='ij') 
    # def dir_matrix(self,x):
    #     dx=x[1]-x[0]
    #     Nx = len(x[1:-1])
    #     kernel = np.zeros((Nx, 1))
    #     kernel[-1] = 1.
    #     kernel[0] = -2.
    #     kernel[1] = 1.
    #     D2 = circulant(kernel)
    #     D2[0, -1] = 0.
    #     D2[-1, 0] = 0.
    #     return D2/dx/dx   
# def solve_subdomain(x,y,F,bc,l,side):
#     # X, Y = np.meshgrid(x[1:], y[1:-1], indexing='ij') 
#     # bc=np.cos(0.5)*np.sin(math.pi*y[1:-1])
#     # u=np.sin(X)*np.sin(math.pi*Y)
#     solver=rect_solver(x,y,l,side,bc)
#     M=solver.calc_D()
#     term=Constants.k* scipy.sparse.identity(M.shape[0])
#     G=solver.calc_bc()
#     # print((M@(u.flatten())+G-(-u-math.pi**2*u).flatten()).reshape((len(x)-1,len(y)-2)))
#     return scipy.sparse.linalg.spsolve(M+term, -G+F)

# class rect_solver:
#     def __init__(self,x,y,l,where,robin=None, side=1):
#         self.x=x
#         self.side=side
#         self.robin=robin
#         self.dx=x[1]-x[0]
#         self.dy=y[1]-y[0]
#         self.y=y
#         self.l=l
#         self.where=where
#         if self.where:
#             self.X, self.Y = np.meshgrid(x[1:], y[1:-1], indexing='ij')
#         else:
#             self.X, self.Y = np.meshgrid(x[:-1], y[1:-1], indexing='ij')
            
#     def  calc_D_x(self):   
#         Nx = len(self.x[:-1])

        
#         kernel = np.zeros((Nx, 1))
#         kernel[-1] = 1.
#         kernel[0] = -2.
#         kernel[1] = 1.
#         D2 = circulant(kernel).astype(complex)
#         D2[0, -1] = 0.
#         D2[-1, 0] = 0.
#         if self.where:
#             D2[-1,-1]=-2-2*self.dx*self.l
#             D2[-1,-2]=2
#         else:    
#             D2[0,0]=-2-2*self.dx*self.l
#             D2[0,1]=2

#         return D2/self.dx/self.dx
    
#     def  calc_D_y(self):   

#         Ny= len(self.y[1:-1])
        
#         kernel = np.zeros((Ny, 1))
#         kernel[-1] = 1.
#         kernel[0] = -2.
#         kernel[1] = 1.
#         D2 = circulant(kernel).astype(complex)
#         D2[0, -1] = 0.
#         D2[-1, 0] = 0.
#         return D2/self.dy/self.dy
    
#     def calc_D(self):
#         return csr_matrix(kron(self.calc_D_x(), identity(len(self.y)-2)),dtype=np.cfloat)+csr_matrix(kron(identity(len(self.x)-1), self.calc_D_y()), dtype=np.cfloat)

    
#     def calc_bc(self):
#         BC=(self.X*0).astype(complex)
#         for i in range(len(self.x[1:])):
#             for j in range(len(self.y[1:-1])):
#                 if self.where:
#                     if abs(self.X[i,j]-self.x[-1])<1e-12:
#                             BC[i,j]=2*self.robin[j]/(self.dx)
                            
#                 else:   
#                     if abs(self.X[i,j]-self.x[0])<1e-12:
#                         BC[i,j]=-2*self.robin[j]/(self.dx) 
        
#         return BC.flatten()       


 

        
    # def save(self, path):
    #     assert self.is_legit()
    #     data = {
    #         "vertices":self.vertices,
    #         "ev": self.ev,
    #         "principal_ev": self.ev[-1], 
    #         "interior_points": self.interior_points,
    #         "hot_points": self.hot_points,
    #         "hot_indices":self.hot_indices,
    #         # "hot_points": self.hot_points[np.lexsort((self.hot_points[:,1], self.hot_points[:,0]))],
    #         "generators": self.generators,
    #         "M": self.M[self.interior_indices][:, self.interior_indices],
    #         'radial_basis':self.radial_functions,
    #          'angle_fourier':self.fourier_coeff,
    #          'translation':self.T,
    #          'cells':self.cells,
    #          'X':self.X,
    #          'geo':self.geo,
    #          'V':self.V,
    #         "legit": True,
    #         'type': 'polygon'
    #     }
    #     torch.save(data, path)

    

# class Annulus(Polygon):
#     def __init__(self, generators,T):
#         self.generators = generators
#         self.n=self.generators.shape[0]
#         self.geo =dmsh.Rectangle(0, 1, 0, 1)- dmsh.Polygon(self.generators)
#         self.fourier_coeff = self.fourier()
#         self.T=T












 