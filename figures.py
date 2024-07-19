import numpy as np
import matplotlib.pyplot as plt
from constants import Constants
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from utils import generate_grf
from main import generate_domains
path='/Users/idanversano/Documents/project_geo_deeponet/tex/figures/attention_deeponet/'
x0=np.linspace(0,1,15)
y0=np.linspace(0,1,15)
X0,Y0=np.meshgrid(x0,y0,indexing='ij')
X0=X0.flatten()
Y0=Y0.flatten()

x=np.linspace(0,1,15)
X,Y=np.meshgrid(x,x, indexing='ij')
def plot_Lshape(ax2):
    x1=np.linspace(0,1/2,8)
    y1=np.linspace(0,1,15)
    x2=np.linspace(8/14,1,7)
    y2=np.linspace(0,1/2,8)
    X1,Y1=np.meshgrid(x1,y1, indexing='ij')
    X2,Y2=np.meshgrid(x2,y2, indexing='ij')
    ax2.scatter(X1.flatten(), Y1.flatten(), color='black',s=1)
    ax2.scatter(X2.flatten(), Y2.flatten(), color='black',s=1)
    plt.xticks([])
    plt.yticks([])

def fig1():
    # Create a figure and two subplots
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))  # Adjust figsize as needed

    # Plot data on the first subplot
    ax.scatter(X0, Y0, color='black',s=10,facecolors='black', edgecolor='black')  
    ax.set_title('Train domain, N=15')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    # Adjust layout and save the figure
    plt.tight_layout()  # Adjust subplot spacing

    plt.savefig(path+'fig1.eps', format='eps', bbox_inches='tight', dpi=600) # Save the figure as an image file

    # Show the plots
    plt.show()


def fig2():
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4)) 
    data=torch.load(Constants.outputs_path+'output1.pt')
    for i in range(1,len(data['err'])):
        ax1.scatter(i,np.log(data['err'][i]),color='black', s=1)
    left, bottom, width, height = [0.2, 0.2, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])  
    ax2.scatter(X0, Y0, color='black',s=10,facecolors='none', edgecolor='black')  
    ax2.scatter(data['X'], data['Y'], color='black',s=10)  
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax1.set_xlabel('iter.')
    ax1.set_ylabel('log-err')
    ax1.axis('tight')    
    ax1.set_title('J='+str(data['J'])+', '+'alpha='+str(data['alpha']))
    plt.tight_layout() 
    plt.savefig(path+'fig2.eps', format='eps', bbox_inches='tight', dpi=600)  # Save the figure as an image file

    plt.show() 

def fig3():
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4)) 
    data=torch.load(Constants.outputs_path+'output2.pt')
    for i in range(1,len(data['err'])):
        ax1.scatter(i,np.log(data['err'][i]),color='black', s=1)
    ax1.set_title('J='+str(data['J'])+', '+'alpha='+str(data['alpha']))    
    left, bottom, width, height = [0.2, 0.2, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])  
    ax2.scatter(X0, Y0, color='red',s=10,facecolors='none', edgecolor='black')  
    ax2.scatter(data['X'], data['Y'], color='black',s=10)  
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax1.set_xlabel('iter.')
    ax1.set_ylabel('log-err')
    ax1.axis('tight')    
    plt.tight_layout() 
    plt.savefig(path+'fig3.eps', format='eps', bbox_inches='tight', dpi=600)  # Save the figure as an image file

    plt.show() 
    
def fig4():
    x0=np.linspace(0,1,15)
    y0=np.linspace(0,1,15)
    X0,Y0=np.meshgrid(x0,y0,indexing='ij')
    X0=X0.flatten()
    Y0=Y0.flatten()
    # Create a figure and two subplots
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))  # Adjust figsize as needed

    # Plot data on the first subplot
    ax.scatter(X0, Y0, color='black',s=20,facecolors='black', edgecolor='black')  
    # ax.set_title('Train domain')
    ax.set_xticks([])
    ax.set_yticks([])
    # Adjust layout and save the figure
    plt.tight_layout()  # Adjust subplot spacing

    plt.savefig(path+'fig4.eps', format='eps', bbox_inches='tight', dpi=600) # Save the figure as an image file

    # Show the plots
    plt.show()
        
def fig5():
    fig, ax = plt.subplots(1, 1, figsize=(2, 2)) 
    data=torch.load(Constants.outputs_path+'output1.pt')
    ax.scatter(X0, Y0, color='black',s=20,facecolors='none', edgecolor='black')  
    ax.scatter(data['X'], data['Y'], color='black',s=10)  
    ax.set_xticks([])
    ax.set_yticks([]) 
    # ax.set_title('Train domain')
    ax.set_xticks([])
    ax.set_yticks([])
    # Adjust layout and save the figure
    plt.tight_layout()  # Adjust subplot spacing

    plt.savefig(path+'fig5.eps', format='eps', bbox_inches='tight', dpi=600) # Save the figure as an image file

    # Show the plots
    plt.show()
    
def fig6():
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(8, 4)) 
    data=torch.load(Constants.outputs_path+'output3.pt')
    ax1.scatter(X0, Y0, color='black',s=20,facecolors='none', edgecolor='black')  
    ax1.scatter(data['X'], data['Y'], color='black',s=10)  
    ax1.set_xticks([])
    ax1.set_yticks([]) 
    # ax.set_title('Train domain')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('N=30')
    for i in range(1,len(data['err'])):
        ax2.scatter(i,np.log(data['err'][i]),color='black', s=1)
    ax2.set_xlabel('iter.')
    ax2.set_ylabel('log-err')
    ax1.axis('tight')  
    ax2.axis('tight')    
    # ax2.set_title('J='+str(data['J'])+', '+'alpha='+str(data['alpha']))        
    
    # Adjust layout and save the figure
    plt.tight_layout()  # Adjust subplot spacing

    plt.savefig(path+'fig6.eps', format='eps', bbox_inches='tight', dpi=600) # Save the figure as an image file

    # Show the plots
    plt.show()
    
def fig7():
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4)) 
    datags=torch.load(Constants.outputs_path+'output4.pt')
    data=torch.load(Constants.outputs_path+'output5.pt')
    for i in range(1,len(data['err'])):
        ax1.scatter(i,np.log(data['err'][i]),color='black', s=1)
        ax1.scatter(i,np.log(datags['err'][i]),color='red', s=1)
    ax1.text(300, -15, 'Hints', fontsize=10, color='black')
    ax1.text(5, 0.5, 'GS', fontsize=10, color='red')    
    left, bottom, width, height = [0.2, 0.2, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])  
    ax2.scatter(X0, Y0, color='black',s=10,facecolors='none', edgecolor='black')  
    ax2.scatter(data['X'], data['Y'], color='black',s=10)  
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax1.set_xlabel('iter.')
    ax1.set_ylabel('log-err')
    ax1.axis('tight')    
    ax1.set_title('J='+str(data['J'])+', '+'alpha='+str(data['alpha']))
    plt.tight_layout() 
    plt.savefig(path+'fig7.eps', format='eps', bbox_inches='tight', dpi=600)  # Save the figure as an image file

    plt.show() 

def fig8():
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4)) 
    datags=torch.load(Constants.outputs_path+'output6.pt')
    data=torch.load(Constants.outputs_path+'output7.pt')
    for i in range(1,len(data['err'])):
        ax1.scatter(i,np.log(data['err'][i]),color='black', s=1)
        ax1.scatter(i,np.log(datags['err'][i]),color='red', s=1)
    ax1.text(300, -10, 'Hints', fontsize=10, color='black')
    ax1.text(30, 0, 'GS', fontsize=10, color='red')    
    left, bottom, width, height = [0.2, 0.2, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])  
    ax2.scatter(X0, Y0, color='black',s=10,facecolors='none', edgecolor='black')  
    ax2.scatter(data['X'], data['Y'], color='black',s=10)  
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax1.set_xlabel('iter.')
    ax1.set_ylabel('log-err')
    ax1.axis('tight')    
    ax1.set_title('J='+str(data['J'])+', '+'alpha='+str(data['alpha']))
    plt.tight_layout() 
    plt.savefig(path+'fig8.eps', format='eps', bbox_inches='tight', dpi=600)  # Save the figure as an image file

    plt.show() 

def fig9():
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4)) 
    datags=torch.load(Constants.outputs_path+'output8.pt')
    data=torch.load(Constants.outputs_path+'output9.pt')
    for i in range(1,len(data['err'])):
        ax1.scatter(i,np.log(data['err'][i]),color='black', s=1)
        ax1.scatter(i,np.log(datags['err'][i]),color='red', s=1)
    ax1.text(300, -20, 'Hints', fontsize=10, color='black')
    ax1.text(200, -8, 'GS', fontsize=10, color='red')    
    left, bottom, width, height = [0.2, 0.2, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])  
    ax2.scatter(X0, Y0, color='black',s=10,facecolors='none', edgecolor='black')  
    ax2.scatter(data['X'], data['Y'], color='black',s=10)  
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax1.set_xlabel('iter.')
    ax1.set_ylabel('log-err')
    ax1.axis('tight')    
    ax1.set_title('J='+str(data['J'])+', '+'alpha='+str(data['alpha']))
    plt.tight_layout() 
    plt.savefig(path+'fig9.eps', format='eps', bbox_inches='tight', dpi=600)  # Save the figure as an image file

    plt.show() 

def fig10():
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4)) 
    datags=torch.load(Constants.outputs_path+'output10.pt')
    data=torch.load(Constants.outputs_path+'output11.pt')
    for i in range(1,len(data['err'])):
        ax1.scatter(i,np.log(data['err'][i]),color='black', s=1)
        ax1.scatter(i,np.log(datags['err'][i]),color='red', s=1)
    ax1.text(200, -6, 'Hints', fontsize=10, color='black')
    ax1.text(100, 2, 'GS', fontsize=10, color='red')    
    left, bottom, width, height = [0.2, 0.2, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])  
    ax2.scatter(X0, Y0, color='black',s=10,facecolors='none', edgecolor='black')  
    ax2.scatter(data['X'], data['Y'], color='black',s=10)  
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax1.set_xlabel('iter.')
    ax1.set_ylabel('log-err')
    ax1.axis('tight')    
    ax1.set_title('J='+str(data['J'])+', '+'alpha='+str(data['alpha']))
    plt.tight_layout() 
    plt.savefig(path+'fig10.eps', format='eps', bbox_inches='tight', dpi=600)  # Save the figure as an image file

    plt.show() 
    
def fig11():
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(8, 4)) 
    spec_single=torch.load(Constants.outputs_path+'spec_single.pt')        
    spec_mult=torch.load(Constants.outputs_path+'spec_mult.pt')

    ax1.plot(np.log(spec_single['low'][:20]),color='black', label=r'$ \mathcal{N}$')
    ax1.plot(np.log(spec_mult['low'])[:20],color='red', linestyle='dashed', label= r'$ \mathcal{N}_{mult}$')
    
    ax2.plot(np.log(spec_single['high'][:20]),color='black', label=r'$ \mathcal{N}$')
    ax2.plot(np.log(spec_mult['high'])[:20],color='red', linestyle='dashed',label=r'$ \mathcal{N}_{mult}$')
    ax1.legend()
    ax2.legend()
    ax2.set_title('smallest mode')
    ax1.set_title('highest mode')
    ax1.set_xlabel('iter.')
    ax1.set_ylabel('log-err')
    ax2.set_xlabel('iter.')
    plt.suptitle('Error of Hints by modes', fontsize=16)
    plt.tight_layout() 
    plt.savefig(path+'fig11.eps', format='eps', bbox_inches='tight', dpi=600)  # Save the figure as an image file

    plt.show() 






# plt.plot(spec_single['low'][:40])  
# plt.plot(spec_mult['low'][:40],color='red')
# plt.show()