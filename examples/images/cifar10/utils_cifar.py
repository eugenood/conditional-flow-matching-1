import copy

import torch
import torchsde
from torchdyn.core import NeuralODE, NeuralSDE

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def generate_samples(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()

def generate_samples_stoch(model, parallel, savedir, step, sigma, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    node_ = NeuralSDE(model_, lambda t, x: sigma, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()

# def generate_samples_sde(model, parallel, savedir, step, sigma, net_="normal"):
#     """Save 64 generated images (8 x 8) for sanity check along training.

#     Parameters
#     ----------
#     model:
#         represents the neural network that we want to generate samples from
#     parallel: bool
#         represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
#     savedir: str
#         represents the path where we want to save the generated images
#     step: int
#         represents the current step of training
#     """
#     model.eval()

#     model_ = copy.deepcopy(model)
#     if parallel:
#         # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
#         model_ = model_.module.to(device)

#     from torchcfm.utils import torch_sde_wrapper2
#     sde = torch_sde_wrapper2(model, (3, 32, 32), sigma).to(device)
#     with torch.no_grad():
#         x0 = torch.randn(64, 3 * 32 * 32, device=device)
#         t_span = torch.linspace(0, 1, 100, device=device)
#         traj = torchsde.sdeint(sde, x0, t_span)
#         traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
#         traj = traj / 2 + 0.5


#     save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

#     model.train()

def generate_samples_sde(model, parallel, savedir, step, sigma, net_="normal"):
    model.eval()
    integration_steps = 100
    with torch.no_grad():
        dt = torch.tensor(1 / integration_steps, device=device)
        t_span = torch.linspace(0, 1, integration_steps + 1, device=device)
        x = torch.randn(64, 3, 32, 32, device=device)
        dW = torch.sqrt(dt) * torch.randn(integration_steps, 64, 3, 32, 32, device=device)
        for i in range(integration_steps):
            f = model(t_span[i], x)
            g = sigma
            x = x + f * dt + g * dW[i]
        x = x.clip(-1, 1)
        save_image(x[:64] / 2 + 0.5, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)
    model.train()



def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x
