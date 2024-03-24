import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import torch
import torchdyn
from torchdyn.datasets import generate_moons

# Implement some helper functions


def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))

class torch_sde_wrapper(torch.nn.Module):
    """Wraps model to torchsde compatible format."""
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self, u, dim, sigma):
        super().__init__()
        self.u = u
        self.dim = dim
        self.sigma = sigma

    def f(self, t, x):
        x = x.reshape(-1, self.dim[0], self.dim[1], self.dim[2])
        out = self.u(t, x)
        out = out.reshape(out.shape[0], -1)
        return out
        # return self.u(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))

    def g(self, t, x):
        cov = self.sigma * torch.eye(self.dim[0] * self.dim[1] * self.dim[2]).to(x.device)
        # cov = cov.reshape((1, self.dim, self.dim))
        cov = cov.repeat(x.shape[0], 1, 1)
        return cov

class torch_sde_wrapper2(torch.nn.Module):
    """Wraps model to torchsde compatible format."""
    noise_type = 'diagonal'
    # noise_type = 'scalar'
    sde_type = 'ito'

    def __init__(self, u, dim, sigma):
        super().__init__()
        self.u = u
        self.dim = dim
        self.sigma = sigma
        # self.iter = 0

    # @torch.no_grad
    # def f(self, t, x):
    #     # print("f", self.iter)
    #     self.iter += 1
    #     x = x.reshape(-1, self.dim[0], self.dim[1], self.dim[2])
    #     out = self.u(t, x)
    #     out = out.reshape(out.shape[0], -1)
    #     return out
    #     # return self.u(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))

    # @torch.no_grad
    # def g(self, t, x):
    #     # print("g", self.iter)
    #     self.iter += 1
    #     # cov = self.sigma * torch.eye(self.dim[0] * self.dim[1] * self.dim[2]).to(x.device)
    #     cov = self.sigma * torch.ones(self.dim[0] * self.dim[1] * self.dim[2]).to(x.device)
    #     cov = cov.repeat(x.shape[0], 1)
    #     # cov = cov.repeat(x.shape[0], 1, 1)
    #     # cov = cov.unsqueeze(2)
    #     # print(self.iter)
    #     # self.iter += 1
    #     return cov

    # @torch.no_grad
    # def g_prod(self, t, x, v):
    #     print("gp", self.iter)
    #     self.iter += 1
    #     return self.sigma * v

    @torch.no_grad
    def f_and_g_prod(self, t, x, v):
        x = x.reshape(-1, self.dim[0], self.dim[1], self.dim[2])
        f = self.u(t, x)
        f = f.reshape(f.shape[0], -1)
        g = self.sigma * v
        # g = g.reshape(g.shape[0], -1)
        return f, g
        # cov = self.sigma * torch.eye(self.dim[0] * self.dim[1] * self.dim[2]).to(x.device)
        # cov = self.sigma * torch.ones(self.dim[0] * self.dim[1] * self.dim[2]).to(x.device)
        # cov = cov.repeat(x.shape[0], 1)
        # cov = cov.repeat(x.shape[0], 1, 1)
        # cov = cov.unsqueeze(2)
        # print(self.iter)
        # self.iter += 1
        # return cov

def plot_trajectories(traj):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["x0", "x1"])
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_lined_trajectories(traj):
    """Plot trajectories of some selected samples."""
    fig, ax = plt.subplots(figsize=(6, 6))
    n = traj.shape[1]
    colors = np.arctan(traj[0, :, 1] / traj[0, :, 0])
    ax.plot(traj[[0,-1], :n, 0], traj[[0,-1], :n, 1], alpha=0.05, marker='.')
    ax.plot(traj[[0,-1], :n:25, 0], traj[[0,-1], :n:25, 1], marker='.')
    ax.legend(["x0", "x1"])
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def animate_trajectories(traj):
    """Animate trajectories of some selected samples."""
    fig, ax = plt.subplots()
    n = traj.shape[1]
    ax.set_xlim(np.min(traj[:, :, 0]), np.max(traj[:, :, 0]))
    ax.set_ylim(np.min(traj[:, :, 1]), np.max(traj[:, :, 1]))
    colors = np.arctan(traj[0, :, 1] / traj[0, :, 0])
    sc = ax.scatter(traj[0,:,0], traj[0,:,1], c=colors, cmap='hsv', alpha=0.2, edgecolors='k')
    cbar = plt.colorbar(sc)
    def update(frame):
        if frame < traj.shape[0]:
            sc.set_offsets(traj[frame,:,:])  # Update scatter plot with new positions
        return sc,
    animation = FuncAnimation(fig, update, frames=range(traj.shape[0] + 10), interval=50, blit=True)
    return animation
