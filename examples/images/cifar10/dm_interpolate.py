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
flags.DEFINE_float("sigma", 0.01, help="sigma")
flags.DEFINE_string("sigmastr", "0.01", help="string for the sigma")
flags.DEFINE_integer("integration_steps", 500, help="number of inference steps")
flags.DEFINE_string("integration_method", "euler", help="integration method to use")
flags.DEFINE_integer("step", 400000, help="training steps")
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
state_dict = checkpoint["net_model"]
try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    new_net.load_state_dict(new_state_dict)
new_net.eval()

def gen_1_img(num_images=64):
    with torch.no_grad():
        dt = torch.tensor(1 / FLAGS.integration_steps, device=device)
        t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
        x = torch.randn(num_images, 3, 32, 32, device=device)
        dW = torch.sqrt(dt) * torch.randn(FLAGS.integration_steps, num_images, 3, 32, 32, device=device)
        for i in range(FLAGS.integration_steps):
            f = new_net(t_span[i], x)
            g = FLAGS.sigma
            x = x + f * dt + g * dW[i]
        x = x.clip(-1, 1)
        return x

x = gen_1_img(2)
# save_image(x[:64] / 2 + 0.5, "test.png", nrow=8)

@torch.no_grad
def interpolate(x0, x1):
    Xt = [x0]
    x = x0
    for i in range(100):
        if torch.sum((x1 - x) ** 2) > 0.3:
            x = x + 1e-4 * (x1 - x0)
            for j in range(30):
                grad = new_net(torch.tensor([1.0], device=device), x)
                x = x + 2e-4 * grad
            if i % 10 == 0:
                Xt.append(x)
    Xt = torch.cat(Xt)
    Xt = torch.cat((Xt, x1))
    print(Xt.shape)
    return Xt

# print(x[0].shape)
interp = interpolate(x[0].unsqueeze(0), x[1].unsqueeze(0))
save_image(interp / 2 + 0.5, "test.png", nrow=1)
