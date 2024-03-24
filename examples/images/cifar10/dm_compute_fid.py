# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import os
import sys

import torchsde
import matplotlib.pyplot as plt
import torch
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE, NeuralSDE
from torchcfm.utils import torch_sde_wrapper2
from torchvision.utils import make_grid, save_image

from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("input_dir", "./results", help="output_directory")
flags.DEFINE_string("model", "dm", help="flow matching model type")
flags.DEFINE_float("sigma", 1e-4, help="sigma")
flags.DEFINE_string("sigmastr", "1e-4", help="string for the sigma")
flags.DEFINE_integer("integration_steps", 500, help="number of inference steps")
flags.DEFINE_string("integration_method", "euler", help="integration method to use")
flags.DEFINE_integer("step", 300000, help="training steps")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 256, help="Batch size to compute FID")

FLAGS(sys.argv)


# Define the model
use_cuda = torch.cuda.is_available()
# device = torch.device("cpu" if use_cuda else "cpu")
device = torch.device("cuda" if use_cuda else "cpu")

new_net = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=FLAGS.num_channel,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(device)


# Load the model
PATH = f"{FLAGS.input_dir}/{FLAGS.model}{FLAGS.sigmastr}/{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt"
print("path: ", PATH)
checkpoint = torch.load(PATH, map_location=device)
state_dict = checkpoint["ema_model"]
try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    new_net.load_state_dict(new_state_dict)
new_net.eval()


# # Define the integration method if euler is used
# if FLAGS.integration_method == "euler":
#     node = NeuralODE(new_net, solver=FLAGS.integration_method)

# sde = torch_sde_wrapper2(new_net, (3, 32, 32), FLAGS.sigma).to(device)
# node_ = NeuralSDE(drift_func=new_net, diffusion_func=lambda t, x: FLAGS.sigma, solver=FLAGS.integration_method)

# def gen_1_img(unused_latent):
#     minibatch_size = FLAGS.batch_size_fid
#     num_iters = FLAGS.batch_size_fid // minibatch_size
#     img = torch.zeros((FLAGS.batch_size_fid, 3, 32, 32), device=device, dtype=torch.uint8)
#     with torch.no_grad():
#         for i in range(num_iters):
#             x0 = torch.randn(minibatch_size, 3 * 32 * 32, device=device)
#             t_span = torch.linspace(0, 1, 100, device=device)
#             traj = torchsde.sdeint(sde, x0, t_span, method='euler', adaptive=True)
#             traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
#             trag = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
#             startind = minibatch_size * i
#             img[startind:startind+minibatch_size] = trag
#     print("generated")
#     return img

# def gen_1_img(unused_latent):
#     with torch.no_grad():
#         traj = node_.trajectory(
#             torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device),
#             t_span=torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device),
#         )
#         traj = traj[-1, :]  # .view([-1, 3, 32, 32]).clip(-1, 1)
#         img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)
#         # traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
#         # traj = traj / 2 + 0.5
#     return traj

# def gen_1_img(unused_latent):
#     with torch.no_grad():
#         # x0 = torch.randn(64, 3 * 32 * 32, device=device)
#         x0 = torch.randn(FLAGS.batch_size_fid, 3 * 32 * 32, device=device)
#         t_span = torch.linspace(0, 1, 100, device=device)
#         # traj = torchsde.sdeint(sde, x0, t_span)
#         bm = torchsde.BrownianInterval(t0=0., t1=1., levy_area_approximation='space-time', size=(FLAGS.batch_size_fid, 3 * 32 * 32), device=device)
#         traj = torchsde.sdeint(sde, x0, t_span, method='euler', bm=bm)
#         traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
#         imgtraj = traj / 2 + 0.5
#         save_image(imgtraj[:64], "test.png", nrow=8)
#         traj = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
#         # print("generated")
#     return traj

def gen_1_img(unused_latent):
    with torch.no_grad():
        dt = torch.tensor(1 / FLAGS.integration_steps, device=device)
        t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
        x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)
        dW = torch.sqrt(dt) * torch.randn(FLAGS.integration_steps, FLAGS.batch_size_fid, 3, 32, 32, device=device)
        for i in range(FLAGS.integration_steps):
            f = new_net(t_span[i], x)
            g = FLAGS.sigma
            x = x + f * dt + g * dW[i]
        x = x.clip(-1, 1)
        # save_image(x[:64] / 2 + 0.5, "test.png", nrow=8)
        return (x * 127.5 + 128).clip(0, 255).to(torch.uint8)

print("Start computing FID")
score = fid.compute_fid(
    gen=gen_1_img,
    dataset_name="cifar10",
    batch_size=FLAGS.batch_size_fid,
    dataset_res=32,
    num_gen=FLAGS.num_gen,
    dataset_split="train",
    mode="legacy_tensorflow",
    device=device
)
print()
print(f"FID for {FLAGS.sigmastr} has been computed")
# print()
# print("Total NFE: ", new_net.nfe)
print()
print("FID: ", score)
