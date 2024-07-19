import numpy as np
import matplotlib.pyplot as plt
from test_deeponet import domain
from utils import *
from scipy.sparse import csr_matrix, kron, identity, lil_matrix
from packages.my_packages import  interpolation_2D, Restriction_matrix
from scipy.sparse import coo_array, bmat
def generate_domains(i1,i2,j1,j2):
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    x_ref=d_ref.x
    y_ref=d_ref.y
    return domain(x_ref[i1:i2], y_ref[j1:j2])

def generate_f_g(shape, seedf):

        f=generate_random_matrix(shape,seed=seedf)
        
        f=(f-np.mean(f))/np.std(f)
        
        
       
        return f

def masking_coordinates(X,Y):
        xx,yy=np.meshgrid(np.linspace(0,1, Constants.n),np.linspace(0,1, Constants.n),indexing='ij')
        X0=xx.flatten()
        Y0=yy.flatten()
        original_points=[(X0[i],Y0[i]) for i in range(len(X0))]
        points=np.array([(X[i],Y[i]) for i in range(len(X))])
        valid_indices=[]
        masked_indices=[]
        for j,p in enumerate(original_points):
            dist=[np.linalg.norm(np.array(p)-points[i]) for i in range(points.shape[0])]
            if np.min(dist)<1e-14:
                valid_indices.append(j)
            else:
                masked_indices.append(j)    
        return valid_indices, masked_indices     
        
def generate_example(sigma=0.1,l=0.2,mean=0):  
    x1=np.linspace(0,1/2,8) 
    y1=np.linspace(0,1,15)
    X1,Y1=np.meshgrid(x1,y1,indexing='ij')
    X1,Y1=X1.flatten(), Y1.flatten()

    x2=np.linspace(8/14,1,7) 
    y2=np.linspace(0,1/2,8)
    X2,Y2=np.meshgrid(x2,y2,indexing='ij')
    X2,Y2=X2.flatten(), Y2.flatten()
    d1=domain(x1,y1)
    d2=domain(x2,y2)
    D1=d1.D.todense()
    D2=d2.D.todense()
    D=block_matrix(D1,D2)

    # k=0
    # for i in range(len(X1)):
    #     plt.scatter(X1[i],Y1[i])
    #     plt.text(X1[i],Y1[i],str(k))
    #     k+=1
                
    # for i in range(len(X2)):
    #     plt.scatter(X2[i],Y2[i])
    #     plt.text(X2[i],Y2[i],str(k))
    #     k+=1
    # plt.show()    

    intersection_indices_l=[105,106,107,108,109,110,111,112]
    l_jump=-15
    r_jump=15

    dx=(x1[1]-x1[0])
    for c in intersection_indices_l[1:]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
    D[105,105]=-4/dx/dx-2/dx*Constants.l
    D[105,90]=1/dx/dx
    D[105,120]=1/dx/dx
    D[105,106]=2/dx/dx

    intersection_indices_r=[120,121,122,123,124,125,126,127]
    l_jump=-15
    r_jump=8

    dx=(x1[1]-x1[0])
    for c in intersection_indices_r[1:-1]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
        
    D[120,120]=-4/dx/dx-2/dx*Constants.l
    D[120,120+l_jump]=1/dx/dx
    D[120,120+r_jump]=1/dx/dx
    D[120,121]=2/dx/dx

    D[127,127]=-4/dx/dx-2/dx*Constants.l
    D[127,127+l_jump]=1/dx/dx
    D[127,127+r_jump]=1/dx/dx
    D[127,126]=2/dx/dx


    X,Y=np.concatenate([X1,X2]), np.concatenate([Y1,Y2])

    valid_indices, non_valid_indices=masking_coordinates(X, Y)     
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    f_ref=np.zeros(d_ref.nx*d_ref.ny)
    mask = np.zeros((len(f_ref),len(f_ref)))
    mask[:, non_valid_indices] = float('-inf')  
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)
    
    # f=generate_f_g(len(X), 1)
    f=generate_grf(X,Y,1,sigma,l,mean)[0]

    f_ref[valid_indices]=f
    # f_ref=torch.tensor(f_ref, dtype=torch.float32)
    
    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),dom,mask, X,Y, X,Y,valid_indices

def generate_rect(N):
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)
    # domain with low resolution:
    d=generate_domains(0,8, 0,15)
    X_ref,Y_ref=d.X, d.Y
    mask = np.zeros((Constants.n**2,Constants.n**2))
    mask[:, d.non_valid_indices] = float('-inf') 
    # domain with hugh resolution:
    # N=Constants.n*2-1
    new_domain=domain(np.linspace(d.x[0],d.x[-1],int((N+1)/2)),np.linspace(d.y[0],d.y[-1],N))
    X,Y=new_domain.X, new_domain.Y
    A,G=new_domain.solver()
    #  d.valid_indices are the indices in the reference rectangle which stays inside the new low resolution domain
    
    return A,dom,torch.tensor(mask, dtype=torch.float32), X, Y,X_ref,Y_ref, d.valid_indices

def generate_rect2(N):
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)
    # domain with low resolution:
    d=generate_domains(0,15, 0,15)
    X_ref,Y_ref=d.X, d.Y
    mask = np.zeros((Constants.n**2,Constants.n**2))
    mask[:, d.non_valid_indices] = float('-inf') 
    # domain with hugh resolution:
    # N=Constants.n*2-1
    new_domain=domain(np.linspace(0,1,N),np.linspace(0,1,N))
    X,Y=new_domain.X, new_domain.Y
    A,G=new_domain.solver(X.reshape((N,N)))
    #  d.valid_indices are the indices in the reference rectangle which stays inside the new low resolution domain
    
    return A,dom,torch.tensor(mask, dtype=torch.float32), X, Y,X_ref,Y_ref, d.valid_indices


def generate_example_2(N=29):  
    # X_ref, Y_ref domain points contained in the referwbce domain
    x1=np.linspace(0,1/2,8) 
    y1=np.linspace(0,1,15)
    x2=np.linspace(8/14,1,7) 
    y2=np.linspace(0,1/2,8)

    d1=domain(x1,y1)
    d2=domain(x2,y2)
    X_ref,Y_ref=np.concatenate([d1.X,d2.X]), np.concatenate([d1.Y,d2.Y])
    
    
    
    # d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    # f_ref=np.zeros(d_ref.nx*d_ref.ny)
    x=np.linspace(0,1,N)
    y=np.linspace(0,1,N)
    x1=np.linspace(x[0],x[int((N-1)/2)],int((N+1)/2)) 
    y1=np.linspace(y[0],y[-1],N)
    X1,Y1=np.meshgrid(x1,y1,indexing='ij')
    X1,Y1=X1.flatten(), Y1.flatten()

    x2=np.linspace(x[int((N+1)/2)],x[-1],int((N-1)/2)) 
    y2=np.linspace(y[0],y[int((N-1)/2)],int((N+1)/2))
    X2,Y2=np.meshgrid(x2,y2,indexing='ij')
    X2,Y2=X2.flatten(), Y2.flatten()
    
    
    

    
    
    d1=domain(x1,y1)
    d2=domain(x2,y2)
    D=lil_matrix(bmat([[d1.D, None], [None, d2.D]]))
    # D1=d1.D.todense()
    # D2=d2.D.todense()
    # D=block_matrix(D1,D2)
    
    
    
    
    # k=0
    # for i in range(len(X1)):
    #     plt.scatter(X1[i],Y1[i])
    #     # plt.text(X1[i],Y1[i],str(k))
    #     k+=1
                
    # for i in range(len(X2)):
    #     plt.scatter(X2[i],Y2[i])
    #     plt.text(X2[i],Y2[i],str(k))
    #     k+=1
    # plt.show()    

    intersection_indices_l=[int((N-1)/2*N)+i for i in range(int((N+1)/2))]
    l_jump=-N
    r_jump=N

    dx=(x1[1]-x1[0])
    for c in intersection_indices_l[1:]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
    D[int((N-1)/2*N),int((N-1)/2*N)]=-4/dx/dx-2/dx*Constants.l
    D[int((N-1)/2*N),int((N-1)/2*N)+N]=1/dx/dx
    D[int((N-1)/2*N),int((N-1)/2*N)-N]=1/dx/dx
    D[int((N-1)/2*N),int((N-1)/2*N)+1]=2/dx/dx

    intersection_indices_r=[int((N+1)/2*N)+i for i in range(int((N+1)/2))]
    l_jump=-N
    r_jump=int((N+1)/2)

    dx=(x1[1]-x1[0])
    for c in intersection_indices_r[1:-1]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
        
    D[int((N+1)/2*N),int((N+1)/2*N)]=-4/dx/dx-2/dx*Constants.l
    D[int((N+1)/2*N),int((N+1)/2*N)+l_jump]=1/dx/dx
    D[int((N+1)/2*N),int((N+1)/2*N)+r_jump]=1/dx/dx
    D[int((N+1)/2*N),int((N+1)/2*N)+1]=2/dx/dx

    p=int((N+1)/2*N)+int((N-1)/2)
    D[p,p]=-4/dx/dx-2/dx*Constants.l
    D[p,p+l_jump]=1/dx/dx
    D[p,p+r_jump]=1/dx/dx
    D[p,p-1]=2/dx/dx

        

    X,Y=np.concatenate([X1,X2]), np.concatenate([Y1,Y2])

    valid_indices, non_valid_indices=masking_coordinates(X, Y)     
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    f_ref=np.zeros(d_ref.nx*d_ref.ny)
    mask = np.zeros((len(f_ref),len(f_ref)))
    mask[:, non_valid_indices] = float('-inf')  
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)

  
    
    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),dom,mask, X,Y, X_ref, Y_ref, valid_indices


def generate_obstacle():  
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    obs=generate_domains(3,11, 9,12)
    
    
    X_ref=[]
    Y_ref=[]
    good_ind=[]
    for i in range(len(d_ref.X)):
        dist=[abs(d_ref.X[i]-obs.X[j])+abs(d_ref.Y[i]-obs.Y[j]) for j in range(len(obs.X))]
        if np.min(dist)>1e-10:
            X_ref.append(d_ref.X[i])
            Y_ref.append(d_ref.Y[i])
            good_ind.append(i)   
    D=lil_matrix(d_ref.D)[good_ind,:][:,good_ind]
    valid_indices, non_valid_indices=masking_coordinates(X_ref, Y_ref) 
    f_ref=np.zeros(d_ref.nx*d_ref.ny)
    mask = np.zeros((len(f_ref),len(f_ref)))
    mask[:, non_valid_indices] = float('-inf')  
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)
    f=generate_f_g(len(X_ref), 1)
    func=interpolation_2D(X_ref,Y_ref,f)
    f_ref[valid_indices]=func(X_ref,Y_ref)
    X=X_ref
    Y=Y_ref
    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices


def generate_obstacle2(N):  
    d_out=domain(np.linspace(0,1,N),np.linspace(0,1,N))
    d0=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    
    
    # obs=domain(d_out.x[int (N/4):int (3*N/4)],d_out.y[int (N/4):int (3*N/4)])
    # obs=domain(d_out.x[int (N/10)+1:int (9*N/10)+1],d_out.y[int(N/10)+1:int (9*N/10)+1])
    obs=domain(d_out.x[int (N/4)+1:int (3*N/4)+1],d_out.y[int(N/2)+1:int (7*N/10)+1])
    
    
    X=[]
    Y=[]
    good_ind=[]
    for i in range(len(d_out.X)):
        dist=[abs(d_out.X[i]-obs.X[j])+abs(d_out.Y[i]-obs.Y[j]) for j in range(len(obs.X))]
        if np.min(dist)>1e-10:
            X.append(d_out.X[i])
            Y.append(d_out.Y[i])
            good_ind.append(i)   
    D=lil_matrix(d_out.D)[good_ind,:][:,good_ind]
    valid_indices, non_valid_indices=masking_coordinates(X, Y) 
    
    f_ref=np.zeros(d0.nx*d0.ny)
    mask = np.zeros((len(f_ref),len(f_ref)))
    mask[:, non_valid_indices] = float('-inf')  
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d0.X.reshape(-1, 1), d0.Y.reshape(-1, 1))), dtype=torch.float32)
    

    
    X_ref=[]
    Y_ref=[]
    for i in range(len(d0.X)):
        dist=[abs(d0.X[i]-obs.X[j])+abs(d0.Y[i]-obs.Y[j]) for j in range(len(obs.X))]
        if np.min(dist)>1e-10:
            X_ref.append(d0.X[i])
            Y_ref.append(d0.Y[i])
            

    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),dom,mask, X,Y, X_ref, Y_ref, valid_indices













def generate_two_obstacles(N=29):  
    d0=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    d_out=domain(np.linspace(0,1,N),np.linspace(0,1,N))
    # obs=domain(d_out.x[int (N/4):int (3*N/4)],d_out.y[int (N/4):int (3*N/4)])
    # obs=domain(d_out.x[int (N/10)+1:int (9*N/10)+1],d_out.y[int(N/10)+1:int (9*N/10)+1])
    obs1=domain(d_out.x[int (1*N/5)+1:int (2*N/5)+1],d_out.y[int(2*N/4)+1:int (3*N/4)+1])
    obs2=domain(d_out.x[int (3*N/5)+1:int (4*N/5)+1],d_out.y[int(2*N/4)+1:int (3*N/4)+1])
    
    
    X=[]
    Y=[]
    good_ind=[]
    for i in range(len(d_out.X)):
        dist1=[abs(d_out.X[i]-obs1.X[j])+abs(d_out.Y[i]-obs1.Y[j]) for j in range(len(obs1.X))]
        dist2=[abs(d_out.X[i]-obs2.X[j])+abs(d_out.Y[i]-obs2.Y[j]) for j in range(len(obs2.X))]
        dist=dist1+dist2
        if np.min(dist)>1e-10:
            X.append(d_out.X[i])
            Y.append(d_out.Y[i])
            good_ind.append(i)   
    D=d_out.D.todense()[good_ind,:][:,good_ind]
    valid_indices, non_valid_indices=masking_coordinates(X, Y) 
    
    f_ref=np.zeros(d0.nx*d0.ny)
    mask = np.zeros((len(f_ref),len(f_ref)))
    mask[:, non_valid_indices] = float('-inf')  
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d0.X.reshape(-1, 1), d0.Y.reshape(-1, 1))), dtype=torch.float32)
    


    X_ref=[]
    Y_ref=[]
    for i in range(len(d0.X)):
        dist1=[abs(d0.X[i]-obs1.X[j])+abs(d0.Y[i]-obs1.Y[j]) for j in range(len(obs1.X))]
        dist2=[abs(d0.X[i]-obs2.X[j])+abs(d0.Y[i]-obs2.Y[j]) for j in range(len(obs2.X))]
        dist=dist1+dist2
        if np.min(dist)>1e-10:
            X_ref.append(d0.X[i])
            Y_ref.append(d0.Y[i])
            


    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),dom,mask, X,Y, X_ref, Y_ref, valid_indices

def generate_Lshape():  
    # X_ref, Y_ref domain points contained in the referwbce domain
    x0=np.linspace(0,1,Constants.n)
    y0=np.linspace(0,1,Constants.n)
    
    x1=np.linspace(x0[0],x0[3],4) 
    y1=np.linspace(0,1,15)
    x2=np.linspace(x0[4],1,13) 
    y2=np.linspace(0,x0[7],8)

    d1=domain(x1,y1)
    d2=domain(x2,y2)
    X_ref,Y_ref=np.concatenate([d1.X,d2.X]), np.concatenate([d1.Y,d2.Y])
    
    
    
    # d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    # f_ref=np.zeros(d_ref.nx*d_ref.ny)
    x=np.linspace(0,1,29)
    y=np.linspace(0,1,29)
    x1=np.linspace(x[0],x[14],15) 
    y1=np.linspace(y[0],y[-1],29)
    X1,Y1=np.meshgrid(x1,y1,indexing='ij')
    X1,Y1=X1.flatten(), Y1.flatten()

    x2=np.linspace(x[15],x[-1],14) 
    y2=np.linspace(y[0],y[14],15)
    X2,Y2=np.meshgrid(x2,y2,indexing='ij')
    X2,Y2=X2.flatten(), Y2.flatten()
    
    d1=domain(x1,y1)
    d2=domain(x2,y2)
    D1=d1.D.todense()
    D2=d2.D.todense()
    D=block_matrix(D1,D2)
    
    
    # k=0
    # for i in range(len(X1)):
    #     plt.scatter(X1[i],Y1[i])
    #     plt.text(X1[i],Y1[i],str(k))
    #     k+=1
                
    # for i in range(len(X2)):
    #     plt.scatter(X2[i],Y2[i])
    #     plt.text(X2[i],Y2[i],str(k))
    #     k+=1
    # plt.show()    

    intersection_indices_l=[406+i for i in range(15)]
    l_jump=-29
    r_jump=29

    dx=(x1[1]-x1[0])
    for c in intersection_indices_l[1:]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
    D[406,406]=-4/dx/dx-2/dx*Constants.l
    D[406,435]=1/dx/dx
    D[406,377]=1/dx/dx
    D[406,407]=2/dx/dx

    intersection_indices_r=[435+i for i in range(15)]
    l_jump=-29
    r_jump=15

    dx=(x1[1]-x1[0])
    for c in intersection_indices_r[1:-1]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
        
    D[435,435]=-4/dx/dx-2/dx*Constants.l
    D[435,435+l_jump]=1/dx/dx
    D[435,435+r_jump]=1/dx/dx
    D[435,436]=2/dx/dx

    D[449,449]=-4/dx/dx-2/dx*Constants.l
    D[449,449+l_jump]=1/dx/dx
    D[449,449+r_jump]=1/dx/dx
    D[449,448]=2/dx/dx

        

    X,Y=np.concatenate([X1,X2]), np.concatenate([Y1,Y2])

    valid_indices, non_valid_indices=masking_coordinates(X, Y)     
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    f_ref=np.zeros(d_ref.nx*d_ref.ny)
    mask = np.zeros((len(f_ref),len(f_ref)))
    mask[:, non_valid_indices] = float('-inf')  
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)
    f=generate_f_g(len(X), 1)
    func=interpolation_2D(X,Y,f)
    f_ref[valid_indices]=func(X_ref,Y_ref)
  
    
    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),dom,mask, X,Y, X_ref, Y_ref, valid_indices


def generate_example3(N):
      # X_ref, Y_ref domain points contained in the referwbce domain
    x0=np.linspace(0,1,Constants.n)
    y0=np.linspace(0,1,Constants.n)
    
    x1=np.linspace(x0[0],x0[3],4) 
    y1=np.linspace(0,1,15)
    x2=np.linspace(x0[4],1,11) 
    y2=np.linspace(0,x0[3],4)

    d1=domain(x1,y1)
    d2=domain(x2,y2)
    X_ref,Y_ref=np.concatenate([d1.X,d2.X]), np.concatenate([d1.Y,d2.Y])
    
    
    
    # d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    # f_ref=np.zeros(d_ref.nx*d_ref.ny)
    m1=int((x0[3]-x0[0])*(N-1)+1)
    m2=N-m1
    x=np.linspace(0,1,N)
    y=np.linspace(0,1,N)
    x1=np.linspace(x[0],x[m1-1],m1) 
    y1=np.linspace(0,1,N)
    X1,Y1=np.meshgrid(x1,y1,indexing='ij')
    X1,Y1=X1.flatten(), Y1.flatten()

    x2=np.linspace(x[m1],1,m2) 
    y2=np.linspace(0,x[m1-1],m1)
    X2,Y2=np.meshgrid(x2,y2,indexing='ij')
    X2,Y2=X2.flatten(), Y2.flatten()
    
    
    

    
    
    d1=domain(x1,y1)
    d2=domain(x2,y2)
    D=lil_matrix(bmat([[d1.D, None], [None, d2.D]]))
    # D1=d1.D.todense()
    # D2=d2.D.todense()
    # D=block_matrix(D1,D2)
    
    
    
    
    # k=0
    # for i in range(len(X1)):
    #     plt.scatter(X1[i],Y1[i])
    #     plt.text(X1[i],Y1[i],str(k))
    #     k+=1
                
    # for i in range(len(X2)):
    #     plt.scatter(X2[i],Y2[i])
    #     plt.text(X2[i],Y2[i],str(k))
    #     k+=1
    # plt.show()    
    
    intersection_indices_l=[int((m1-1)*N)+i for i in range(m1)]
    l_jump=-N
    r_jump=N

    dx=(x1[1]-x1[0])
    for c in intersection_indices_l[1:]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
    D[int((m1-1)*N),int((m1-1)*N)]=-4/dx/dx-2/dx*Constants.l
    D[int((m1-1)*N),int((m1-1)*N)+r_jump]=1/dx/dx
    D[int((m1-1)*N),int((m1-1)*N)+l_jump]=1/dx/dx
    D[int((m1-1)*N),int((m1-1)*N)+1]=2/dx/dx

    intersection_indices_r=[int((m1)*N)+i for i in range(m1)]
    l_jump=-N
    r_jump=m1

    dx=(x1[1]-x1[0])
    for c in intersection_indices_r[1:-1]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
        
    D[int((m1)*N),int((m1)*N)]=-4/dx/dx-2/dx*Constants.l
    D[int((m1)*N),int((m1)*N)+l_jump]=1/dx/dx
    D[int((m1)*N),int((m1)*N)+r_jump]=1/dx/dx
    D[int((m1)*N),int((m1)*N)+1]=2/dx/dx

    p=intersection_indices_r[-1]
    D[p,p]=-4/dx/dx-2/dx*Constants.l
    D[p,p+l_jump]=1/dx/dx
    D[p,p+r_jump]=1/dx/dx
    D[p,p-1]=2/dx/dx

        

    X,Y=np.concatenate([X1,X2]), np.concatenate([Y1,Y2])

    valid_indices, non_valid_indices=masking_coordinates(X, Y)     
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    f_ref=np.zeros(d_ref.nx*d_ref.ny)
    mask = np.zeros((len(f_ref),len(f_ref)))
    mask[:, non_valid_indices] = float('-inf')  
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)

  
    
    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),dom,mask, X,Y, X_ref, Y_ref, valid_indices

    # plt.scatter(d_ref.X,d_ref.Y, color='black')    
    # plt.scatter(X_ref,Y_ref, color='red') 
    # plt.show() 
  

        
    # k=0
    # for i in range(len(X_ref)):
    #     plt.scatter(X_ref[i],Y_ref[i])
    #     plt.text(X_ref[i],Y_ref[i],str(k))
    #     k+=1
                

    # plt.show()   
    
    

# A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_example_2()


# generate_rect2(14)     
# D,f,dom,mask=generate_example()

# A, f_ref,f,dom,mask, X, Y, valid_indices =generate_rect2(30)
# print(mask.shape)
# print(f_ref.shape)
