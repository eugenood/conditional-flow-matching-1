"""Implements Conditional Diffusion Matcher Losses."""

# Author: Alex Tong
#         Eugene Lim
#         Kilian Fatras
#         +++
# License: MIT License

import math
import warnings
from typing import Union

import torch

from .optimal_transport import OTPlanSampler


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalDiffusionMatcher:
    """Base class for conditional diffusion matching methods. This class implements the independent
    conditional diffusion matching methods and serves as a parent class for all other diffusion
    matching methods.

    It implements:
    - Drawing data from Brownian bridge probability path N(t * x1 + (1 - t) * x0, t * (1 - t) * I)
    - conditional drift function ut(x|z) = (x1 - x) / (1 - t)
    - score function $\nabla log p_t(x|z)$
    """
    
    def __init__(self, sigma: Union[float, int] = 0.0):
        r"""Initialize the ConditionalDiffusionMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : Union[float, int]
        """
        self.sigma = sigma

    def compute_mean_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, t * (1 - t) * I).

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mean_t: t * x1 + (1 - t) * x0
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_std_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, t * (1 - t) * I).

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation std_t
        """
        return self.sigma * torch.sqrt(t * (1 - t))

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, t * (1 - t) * I).

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)
        """
        mean_t = self.compute_mean_t(x0, x1, t)
        std_t = self.compute_std_t(t)
        std_t = pad_t_like_x(std_t, x0)
        return mean_t + std_t * epsilon

    def compute_conditional_drift(self, x0, x1, t, xt):
        """
        Compute the conditional drift function ut(x|z).

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional drift function ut(x|z)
        """
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - xt) / (1 - t)

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, t * (1 - t) * I))
        and the conditional drift function ut(x|z).

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional drift function ut(x|z)
        (optionally) eps: Tensor, shape (bs, *dim) such that xt = mean_t + std_t * epsilon
        """
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_drift(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut
