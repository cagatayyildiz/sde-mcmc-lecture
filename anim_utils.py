import os, numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib import animation
os.environ['KMP_DUPLICATE_LIB_OK']='True'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


m1   = torch.tensor([0.0,0.0])
std1 = torch.tensor([[1.0,0.8],[0.8,1.0]])
d1   = torch.distributions.MultivariateNormal(m1,std1)

Ngr = 100
w,wplot = 6, 6
xnp = np.linspace(-w, w, Ngr)
ynp = np.linspace(-w, w, Ngr)
X,Y = np.meshgrid(xnp, ynp)
XY  = torch.tensor(np.array([X.T.flatten(), Y.T.flatten()]).T).to(torch.float32)


def plot_surface(loss_fnc, trajs=[], labels=[], colors=['firebrick','darkorange'], styles=['.-','-'], \
                 title=None, return_fig=False):
    if return_fig:
        fig = plt.figure(1,(6,6))
        ax1 = plt.subplot(1,1,1)   
    loss_XY = loss_fnc(XY).detach().numpy().reshape(Ngr,Ngr)
    im = ax1.contour(X, Y, loss_XY, levels=50)
    if title is not None:
        ax1.set_title(title,fontsize=25)
    h1, = ax1.plot(trajs[0][:,0],trajs[0][:,1], styles[0], color=colors[0], lw=3)
    h2, = ax1.plot(trajs[1][:,0],trajs[1][:,1], styles[1], color=colors[1], lw=3)
    if len(labels)>0:
        assert len(labels)==len(trajs)
        ax1.legend(labels,fontsize=18,loc='lower right')
    ax1.tick_params(axis='both',which='both',left=False,right=False,bottom=False,top=False,\
                    labelbottom=False,labelleft=False) 
    plt.tight_layout()
    if return_fig:
        return fig,ax1,h1,h2
    else:
        plt.show()


def plot_opt_animation(loss_fnc, gd_loss_trace, ode_loss_trace):
    num_euler_steps = ode_loss_trace.shape[0] // gd_loss_trace.shape[0]
    num_opt_iters = gd_loss_trace.shape[0] 
    fig,ax1,h1,h2 = plot_surface(loss_fnc, [gd_loss_trace,ode_loss_trace], ['GD','ODE flow'], \
                 title='GD vs ODE flow (iteration - 0)', return_fig=True)
    def animate(i):
        h1.set_data(gd_loss_trace[:i,0],  gd_loss_trace[:i,1])
        h2.set_data(ode_loss_trace[:num_euler_steps*i,0], ode_loss_trace[:num_euler_steps*i,1])
        ax1.set_title(f'GD vs ODE flow (iteration - {i+1})')
        return (ax1,h1,h2)
    anim = animation.FuncAnimation(fig, animate, frames=num_opt_iters, interval=125, blit=False)
    plt.close()
    return anim    


def plot_mcmc_surface(*ps, betas):
    Ngr = 50
    xnp = np.linspace(-2, 6, Ngr)
    ynp = np.linspace(-2, 6, Ngr)
    X,Y = np.meshgrid(xnp, ynp)
    XY  = torch.tensor(np.array([X.T.flatten(), Y.T.flatten()])).to(torch.float32).T
    ds  = [pi_(XY).reshape(Ngr,Ngr) for pi_ in ps]
    L   = len(ps)
    plt.figure(1,(4*L,3))
    for i,dens in enumerate(ds):
        plt.subplot(1,L,i+1)
        plt.contourf(X, Y, dens, levels=20)
        plt.xlim([-2,6])
        plt.ylim([-2,6])
        plt.colorbar()
        plt.title(r'$\beta=${:.2f}'.format(betas[i]),fontsize=18)

def plot_mcmc_animation(pi, thetas, betas, num_frames=20):
    Ngr = 50
    xnp = np.linspace(-2, 6, Ngr)
    ynp = np.linspace(-2, 6, Ngr)
    X,Y = np.meshgrid(xnp, ynp)
    XY  = torch.tensor(np.array([X.T.flatten(), Y.T.flatten()])).to(torch.float32).T
    map_XY = pi(XY).reshape(Ngr,Ngr)
    S = len(betas) // num_frames
    
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14,6))
    gs  = GridSpec(3, 2)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])
    
    def animate(i):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        idx = int(S*i)
        ax1.contourf(X, Y, map_XY, levels=20)
        ax1.plot(thetas[:idx,0], thetas[:idx,1], '-r')
        ax1.plot(thetas[0,0], thetas[0,1], '.w', markersize=20)
        ax1.set_xlim([-2,6])
        ax1.set_ylim([-2,6])
        ax1.set_title(f'samples - iter \# {S*(i+1)}',fontsize=20)
        ax2.plot(np.arange(idx), betas[:idx],'-b')
        ax2.set_xlim([-len(betas)/10, len(betas)])
        ax2.set_ylim([np.min(betas)-np.max(betas)/10, np.max(betas) + np.max(betas)/10])
        ax2.grid()
        ax2.set_title(f'beta - iter \#  {S*(i+1)}',fontsize=15)
        ax3.plot(np.arange(idx), thetas[:idx,0], '-r')
        ax3.set_xlim([-len(betas)/10, len(betas)])
        ax3.set_ylim([np.min(thetas[:,0]) - np.max(thetas[:,0])/10, \
                      np.max(thetas[:,0]) + np.max(thetas[:,0])/10])
        ax3.grid()
        ax4.plot(np.arange(idx), thetas[:idx,1], '-r')
        ax4.set_xlim([-len(betas)/10, len(betas)])
        ax4.set_ylim([np.min(thetas[:,1]) - np.max(thetas[:,1])/10, \
                      np.max(thetas[:,1]) + np.max(thetas[:,1])/10])
        ax4.grid()
        plt.tight_layout()
        
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=125)
    plt.close()
    return anim