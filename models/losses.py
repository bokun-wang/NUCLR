"""
    implementation of other two-way contrastive losses
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class InfoNCE_Loss(nn.Module):
    def __init__(self, world_size=8, temperature=0.01, personalized_tau=False, image_tau=None, text_tau=None):
        super(InfoNCE_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.personalized_tau = personalized_tau
        self.image_tau = image_tau
        self.text_tau = text_tau

    def forward(self, image_features, text_features, image_idx=None, text_idx=None):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        if self.personalized_tau:
            image_temp = self.image_tau[image_idx]
            text_temp = self.text_tau[text_idx]
            sim = torch.einsum('i d, j d -> i j', text_features, image_features)
            labels = torch.arange(image_features.shape[0], device=image_features.device)
            total_loss = (F.cross_entropy(sim / text_temp, labels) + F.cross_entropy(sim.t() / image_temp, labels)) / 2

        else:
            sim = torch.einsum('i d, j d -> i j', text_features, image_features) / self.temperature
            labels = torch.arange(image_features.shape[0], device=image_features.device)
            total_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        return total_loss


class DCL_Loss(nn.Module):
    """
    Paper: https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html
    Code modified from: https://github.com/chingyaoc/DCL
    """

    def __init__(self, world_size=8, bsz=128, temperature=0.01, tau_plus=0.1):
        super(DCL_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.bsz = bsz
        self.mask_neg = (1.0 - torch.eye(bsz)).cuda()  # diagonal: 0; off-diagonal: 1.0

    def forward(self, image_features, text_features, image_idx=None, text_idx=None):
        # Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]

        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        N = batch_size - 1

        pos = torch.exp(diag_sim / self.temperature).squeeze()
        neg_1 = torch.exp(sim / self.temperature) * self.mask_neg
        neg_2 = torch.exp(sim.t() / self.temperature) * self.mask_neg

        Ng_1 = (-self.tau_plus * N * pos + neg_1.sum(dim=-1).squeeze()) / (1 - self.tau_plus)
        Ng_1 = torch.clamp(Ng_1, min=N * np.e ** (-1 / self.temperature))

        Ng_2 = (-self.tau_plus * N * pos + neg_2.sum(dim=-1).squeeze()) / (1 - self.tau_plus)
        Ng_2 = torch.clamp(Ng_2, min=N * np.e ** (-1 / self.temperature))

        total_loss = 0.5 * (-torch.log(pos / (pos + Ng_1))).mean() + 0.5 * (-torch.log(pos / (pos + Ng_2))).mean()
        return total_loss


class GCL_Loss(nn.Module):
    """
    Paper: https://proceedings.mlr.press/v162/yuan22b.html
    Code modified from: https://github.com/zhqiu/contrastive-learning-iSogCLR/blob/c20903bd79d87ff1f99b0af946d92f7c3e3ec3d5/bimodal_exps/models/losses.py#L61
    """

    def __init__(self, N_I, N_T, gamma=0.1, temperature=0.07, world_size=8, bsz=128):
        # Inputs:
        #   N is number of samples in training set

        super(GCL_Loss, self).__init__()
        self.world_size = world_size
        self.s_I = torch.zeros(N_I).cuda()
        self.s_T = torch.zeros(N_T).cuda()
        self.b_I = torch.zeros(N_I).cuda()
        self.b_T = torch.zeros(N_T).cuda()
        self.gamma = gamma
        self.temperature = temperature  # individual temperature not allowed
        self.eps = 1e-10
        self.bsz = bsz
        self.mask_neg = (1.0 - torch.eye(bsz)).cuda()  # diagonal: 0; off-diagonal: 1.0

    def forward(self, image_features, text_features, image_ids, text_ids, epoch):

        # Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]

        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
        text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = torch.exp(
            image_diffs_d_temps - self.b_I[image_ids][:, None]) * self.mask_neg
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * self.mask_neg

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0 - self.gamma) * self.s_I[image_ids] * torch.exp(
                old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
            s_T = (1.0 - self.gamma) * self.s_T[text_ids] * torch.exp(
                old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        weights_image = exp_image_diffs / (s_I + self.eps)
        weights_text = exp_text_diffs / (s_T + self.eps)

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True)

        total_loss = image_loss.mean() + text_loss.mean()

        return total_loss

class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @https://github.com/filipbasara0/simple-clip/blob/main/simple_clip/clip.py
    """

    def __init__(self, world_size, logit_scale, logit_bias, siglip_bidir):
        super(SigLipLoss, self).__init__()
        self.world_size = world_size
        if self.world_size > 1:
            self.rank = dist.get_rank()
        else:
            self.rank = 0
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias
        self.siglip_bidir = siglip_bidir

    def forward(self, image_features, text_features):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        logits = image_features @ text_features.t() * self.logit_scale.exp() + self.logit_bias
        # -1 for off-diagonals and 1 for diagonals
        labels = 2 * torch.eye(logits.size(0), device=logits.device) - 1
        if self.siglip_bidir:
            loss_images = -torch.sum(F.logsigmoid(labels * logits)) / logits.size(0)
            loss_texts = -torch.sum(F.logsigmoid(labels * logits.t())) / logits.size(1)
            total_loss = (loss_images + loss_texts) / 2.0
        else:
            # pairwise sigmoid loss
            total_loss = -torch.sum(F.logsigmoid(labels * logits)) / logits.size(0)

        return total_loss   

def D(z1, z2, mu=1.0):
    mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    z1 = mask1 * z1 + (1-mask1) * F.normalize(z1, dim=1) * np.sqrt(mu)
    z2 = mask2 * z2 + (1-mask2) * F.normalize(z2, dim=1) * np.sqrt(mu)
    loss_part1 = -2 * torch.mean(z1 * z2) * z1.shape[1]
    square_term = torch.matmul(z1, z2.T) ** 2
    loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                 z1.shape[0] / (z1.shape[0] - 1)
    return (loss_part1 + loss_part2) / mu


class Spectral_Loss(nn.Module):
    """
        https://github.com/jhaochenz96/spectral_contrastive_learning/blob/main/models/spectral.py
    """
    def __init__(self, world_size, mu=1.0):
        super(Spectral_Loss, self).__init__()
        self.world_size = world_size
        self.mu = mu

    def forward(self, image_features, text_features):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
        loss = D(image_features, text_features, self.mu)
        return loss


class CyCLIP_Loss(nn.Module):
    """
        https://github.com/goel-shashank/CyCLIP/blob/52d77af2a5f1a4bff01b4c371d6b98e2d0340137/src/train.py
    """

    def __init__(self, world_size, temperature, cylambda_1=0.25, cylambda_2=0.25):
        super(CyCLIP_Loss, self).__init__()

        self.world_size = world_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.cylambda_1 = cylambda_1
        self.cylambda_2 = cylambda_2

    def forward(self, image_features, text_features):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = len(image_features)

        logits_text_per_image = (image_features @ text_features.t()) / self.temperature
        logits_image_per_text = logits_text_per_image.t()

        target = torch.arange(batch_size).long().cuda()

        # contrastive loss, the same as CLIP
        contrastive_loss = (self.criterion(logits_text_per_image, target) + self.criterion(logits_image_per_text,
                                                                                           target)) / 2.0

        # inmodal_cyclic_loss
        logits_image_per_image = (image_features @ image_features.t()) / self.temperature
        logits_text_per_text = (text_features @ text_features.t()) / self.temperature
        inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() * (
                self.temperature ** 2) * batch_size

        # crossmodal_cyclic_loss
        crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() * (
                self.temperature ** 2) * batch_size

        loss = contrastive_loss + self.cylambda_1 * inmodal_cyclic_loss + self.cylambda_2 * crossmodal_cyclic_loss

        return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICReg_Loss(nn.Module):
    """
        VICReg
        https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    """

    def __init__(self, world_size, dim_size, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super(VICReg_Loss, self).__init__()

        self.world_size = world_size
        self.dim_size = dim_size
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, image_features, text_features):
        if self.world_size > 1:
            x = torch.cat(GatherLayer.apply(image_features), dim=0)
            y = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = len(x)

        repr_loss = F.mse_loss(x, y)  # invariance term

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2  # variance term

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.dim_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.dim_size)  # covariance term

        loss = (
                self.sim_coeff * repr_loss
                + self.std_coeff * std_loss
                + self.cov_coeff * cov_loss
        )

        return loss


class RGCL_Loss(nn.Module):
    """
    Paper: https://proceedings.mlr.press/v202/qiu23a.html
    Code modified from: https://github.com/xywei00/csce689_iSogCLR/blob/df1027c2bd073dd352c7ea76c68fea087584f914/bimodal_exps/models/losses.py#L163
    """

    def __init__(self, N_I, N_T, gamma=0.8, tau_init=0.01, world_size=8, bsz=128, rho_I=8.0, rho_T=8.0, eta_init=1e-5,
                 beta_u=0.5):

        # Inputs:
        #   N is number of samples in training set
        #   gamma is beta_0
        #   eta_init is the step size of personalzied temperature

        super(RGCL_Loss, self).__init__()
        self.world_size = world_size
        self.s_I = torch.zeros(N_I).cuda()
        self.s_T = torch.zeros(N_T).cuda()
        self.b_I = torch.zeros(N_I).cuda()
        self.b_T = torch.zeros(N_T).cuda()
        self.gamma = gamma
        self.eps = 1e-10
        self.bsz = bsz
        self.mask_neg = (1.0 - torch.eye(bsz)).cuda()

        self.tau_min, self.tau_max = 0.005, 0.05

        # rho is DRO radius
        self.rho_I = rho_I
        self.rho_T = rho_T

        self.eta_init = eta_init

        self.beta_u = beta_u
        self.grad_clip = 5.0
        # One tau for each anchor
        self.tau_I = torch.ones(N_I).cuda() * tau_init
        self.tau_T = torch.ones(N_T).cuda() * tau_init
        self.u_I = torch.zeros(N_I).cuda()
        self.u_T = torch.zeros(N_T).cuda()

    def forward(self, image_features, text_features, image_ids, text_ids, epoch):

        # Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]

        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        # generate temperatures
        tau_image = self.tau_I[image_ids]
        tau_text = self.tau_T[text_ids]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / tau_image[:, None]).detach()
        text_diffs_d_temps = (text_diffs / tau_text[None, :]).detach()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = torch.exp(
            image_diffs_d_temps - self.b_I[image_ids][:, None]) * self.mask_neg  # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * self.mask_neg

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            # multiply by exp(old_b - new_b) to make sure s and g are in the same scale
            s_I = (1.0 - self.gamma) * self.s_I[image_ids] * torch.exp(
                old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
            s_T = (1.0 - self.gamma) * self.s_T[text_ids] * torch.exp(
                old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        # The exp(-self.b_I) terms in exp_image_diffs and s_I are cancelled out
        weights_image = exp_image_diffs / (s_I + self.eps)
        weights_text = exp_text_diffs / (s_T + self.eps)

        # It is sum instead of mean because B has already been in s_i
        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True)

        total_loss = image_loss.mean() + text_loss.mean()
        # The s_i here is batch_size - 1 times of the one in the paper because g_I uses sum instead of average
        # Thus, we need to divide by batch_size - 1
        # Moreover, the s_i here contains a multiplicative term exp(-self.b). To recover the real s_i, we need
        # to multiple exp(self.b) such that log (x * exp(self.b)) = log(x) + self.b
        # The last term: it is sum instead of mean because B has already been in s_i
        # Both weights_image and image_diffs_d_temps are not related to self.b_I
        temp_weight_image = torch.log(s_I / (batch_size - 1)) + self.b_I[image_ids][:, None] + self.rho_I - torch.sum(
            weights_image * image_diffs_d_temps, dim=1, keepdim=True)
        temp_weight_text = torch.log(s_T / (batch_size - 1)) + self.b_T[text_ids][None, :] + self.rho_T - torch.sum(
            weights_text * text_diffs_d_temps, dim=0, keepdim=True)

        self.u_I[image_ids] = (1.0 - self.beta_u) * self.u_I[
            image_ids] + self.beta_u * temp_weight_image.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)
        self.u_T[text_ids] = (1.0 - self.beta_u) * self.u_T[text_ids] + self.beta_u * temp_weight_text.squeeze().clamp_(
            min=-self.grad_clip, max=self.grad_clip)
        self.tau_I[image_ids] = (tau_image - self.eta_init * self.u_I[image_ids]).clamp_(min=self.tau_min,
                                                                                         max=self.tau_max)
        self.tau_T[text_ids] = (tau_text - self.eta_init * self.u_T[text_ids]).clamp_(min=self.tau_min,
                                                                                      max=self.tau_max)

        return total_loss, tau_image.mean().item(), tau_text.mean().item(), self.eta_init, \
               temp_weight_image.mean().item(), temp_weight_text.mean().item()


class DGCL_Loss(nn.Module):
    def __init__(self, N_I, N_T, gamma, temperature, world_size, bsz, neg_zeta_init, xi_init,
                 theta, start_epochs, eta_init, eta_I_ratio, vanilla):

        super(DGCL_Loss, self).__init__()
        self.N = N_I
        assert N_I == N_T
        self.world_size = world_size
        self.vanilla = vanilla
        self.theta = theta
        self.gamma = gamma
        self.temperature = temperature
        self.start_epochs = start_epochs
        self.xi_init = torch.tensor(xi_init).cuda()
        self.s_I = torch.zeros(N_I).cuda()
        self.s_T = torch.zeros(N_T).cuda()
        self.b_I = torch.zeros(N_I).cuda()
        self.b_T = torch.zeros(N_T).cuda()
        self.z_I = torch.zeros(N_I).cuda()
        self.z_T = torch.zeros(N_T).cuda()
        self.eps = 1e-10
        self.eta_init = eta_init
        self.eta_I_ratio = eta_I_ratio

        self.zeta_I = -neg_zeta_init * torch.ones(N_I).cuda()
        self.xi_I = torch.tensor(xi_init).cuda()
        self.zeta_T = -neg_zeta_init * torch.ones(N_T).cuda()
        self.xi_T = torch.tensor(xi_init).cuda()

        self.bsz = bsz
        self.mask_neg = (1.0 - torch.eye(bsz)).cuda()

    def update_and_get_xi(self):
        """
        set xi to max(xi_init, max(zeta)) at the end of each epoch
        """
        self.xi_I = torch.maximum(torch.max(self.zeta_I).detach().clone(), self.xi_init)
        self.xi_T = torch.maximum(torch.max(self.zeta_T).detach().clone(), self.xi_init)

        return self.xi_I, self.xi_T

    def get_zeta(self):
        """
        return the zeta value
        """
        return self.zeta_I, self.zeta_T

    def forward(self, image_features, text_features, image_ids, text_ids, epoch, max_epoch):

        # Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]

        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]  # add a new axis
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = image_diffs.clone().detach_()
        text_diffs_d_temps = text_diffs.clone().detach_()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        zeta_T_batch = self.zeta_T[text_ids].clone()
        exp_image_diffs = torch.exp((image_diffs_d_temps - self.b_I[image_ids][:, None] - zeta_T_batch[None,
                                                                                          :]) / self.temperature)  # unmasked here

        zeta_I_batch = self.zeta_I[image_ids].clone()
        exp_text_diffs = torch.exp((text_diffs_d_temps - self.b_T[text_ids][None, :] - zeta_I_batch[:,
                                                                                       None]) / self.temperature)  # unmasked here

        g_I = torch.sum(exp_image_diffs * self.mask_neg, dim=1, keepdim=True)  # always masked
        g_T = torch.sum(exp_text_diffs * self.mask_neg, dim=0, keepdim=True)  # always masked

        uninitialized_img_inds = torch.where(self.s_I[image_ids] == 0.0)[0]  # indices within the batch
        uninitialized_txt_inds = torch.where(self.s_T[text_ids] == 0.0)[0]
        assert torch.all(uninitialized_img_inds == uninitialized_txt_inds)

        self.s_I[image_ids[uninitialized_img_inds]] = g_I.squeeze()[uninitialized_img_inds] * torch.exp(
            self.b_I[image_ids[uninitialized_img_inds]] - old_b_I[uninitialized_img_inds])
        self.s_T[text_ids[uninitialized_txt_inds]] = g_T.squeeze()[uninitialized_txt_inds] * torch.exp(
            self.b_T[text_ids[uninitialized_txt_inds]] - old_b_T[uninitialized_txt_inds])

        s_I = (1.0 - self.gamma) * self.s_I[image_ids] * torch.exp(
            old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
        s_T = (1.0 - self.gamma) * self.s_T[text_ids] * torch.exp(
            old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
        s_I = s_I.reshape(g_I.shape)
        s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        # The exp(-self.b_I) terms in exp_image_diffs and s_I are cancelled out
        if self.vanilla:
            offset_I = ((batch_size - 1.0) / (self.N - 1.0)) * torch.exp(
                -(self.zeta_T[text_ids].squeeze() + self.b_I[image_ids]) / self.temperature).squeeze().reshape(s_I.shape)
        else:
            # xi_T = max_j zeta_j of texts
            offset_I = ((batch_size - 1.0) / (self.N - 1.0)) * torch.exp(
                -(self.xi_T + self.b_I[image_ids]) / self.temperature).squeeze().reshape(s_I.shape)
        weights_image = (exp_image_diffs * self.mask_neg) / (s_I + offset_I).clamp(min=1e-16)

        if self.vanilla:
            offset_T = ((batch_size - 1.0) / (self.N - 1.0)) * torch.exp(
                -(self.zeta_I[image_ids].squeeze() + self.b_T[text_ids]) / self.temperature).squeeze().reshape(s_T.shape)
        else:
            offset_T = ((batch_size - 1.0) / (self.N - 1.0)) * torch.exp(
                -(self.xi_I + self.b_T[text_ids]) / self.temperature).squeeze().reshape(s_T.shape)
        weights_text = (exp_text_diffs * self.mask_neg) / (s_T + offset_T).clamp(min=1e-16)

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        if torch.sum(torch.diagonal(weights_image)) > 0.0:
            assert 0, "diag(weights_image) has nonzero value."
        if torch.sum(torch.diagonal(weights_text)) > 0.0:
            assert 0, "diag(weights_text) has nonzero value."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True)  # the diagonal is already 0
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True)

        total_loss = image_loss.mean() + text_loss.mean()

        base_eta = 0.5 * self.eta_init * (1.0 + math.cos(math.pi * (epoch - self.start_epochs) / (max_epoch - 1 - self.start_epochs)))
        if epoch >= self.start_epochs:      # update zeta variables
            if epoch < int(max_epoch / 2):
                cur_eta = base_eta
            elif epoch >= int(max_epoch / 2) and epoch < int(max_epoch * 3 / 4):
                cur_eta = base_eta / 10.0
            else:
                cur_eta = base_eta / 100.0

            cur_eta_I = self.eta_I_ratio * cur_eta
            zeta_I_batch = self.zeta_I[image_ids].clone()
            offset_T_tilde = (1.0 / (self.N - 1.0)) * torch.exp(
                -(self.zeta_I[image_ids].squeeze() + self.b_T[text_ids].squeeze()) / self.temperature).reshape(
                s_T.shape)
            weights_text_tilde = exp_text_diffs / (s_T / (batch_size - 1.0) + offset_T_tilde).clamp(
                min=1e-16)  # numerator always unmasked
            uninitialized_img_inds = torch.where(self.z_I[image_ids] == 0.0)[0]  # indices within the batch
            self.z_I[image_ids[uninitialized_img_inds]] = -(self.N / (self.N - 1.0)) * \
                                                          torch.mean(weights_text_tilde, dim=1)[
                                                              uninitialized_img_inds].squeeze() + 1.0

            self.z_I[image_ids] = (1.0 - self.theta) * self.z_I[image_ids].clone() + self.theta * (
                    -(self.N / (self.N - 1.0)) * torch.mean(weights_text_tilde,
                                                            dim=1) + 1.0)  # average over anchors

            self.zeta_I[image_ids] = zeta_I_batch - cur_eta_I * self.z_I[image_ids]

            cur_eta_T = cur_eta
            zeta_T_batch = self.zeta_T[text_ids].clone()
            offset_I_tilde = (1.0 / (self.N - 1.0)) * torch.exp(
                -(self.zeta_T[text_ids].squeeze() + self.b_I[image_ids].squeeze()) / self.temperature).reshape(
                s_I.shape)
            weights_image_tilde = exp_image_diffs / (s_I / (batch_size - 1.0) + offset_I_tilde).clamp(min=1e-16)
            uninitialized_txt_inds = torch.where(self.z_T[text_ids] == 0.0)[0]  # indices within the batch
            self.z_T[text_ids[uninitialized_txt_inds]] = -(self.N / (self.N - 1.0)) * torch.mean(weights_image_tilde,
                                                                                                 dim=0)[
                uninitialized_txt_inds].squeeze() + 1.0

            self.z_T[text_ids] = (1.0 - self.theta) * self.z_T[text_ids].clone() + self.theta * (
                    -(self.N / (self.N - 1.0)) * torch.mean(weights_image_tilde, dim=0) + 1.0)
            self.zeta_T[text_ids] = zeta_T_batch - cur_eta_T * self.z_T[text_ids]

        else:
            cur_eta_I, cur_eta_T = 0.0, 0.0

        max_zeta_img, avg_zeta_img, min_zeta_img = self.zeta_I[image_ids].max(), self.zeta_I[image_ids].mean(), \
                                                   self.zeta_I[image_ids].min()
        max_zeta_txt, avg_zeta_txt, min_zeta_txt = self.zeta_T[text_ids].max(), self.zeta_T[text_ids].mean(), \
                                                   self.zeta_T[text_ids].min()

        return total_loss, max_zeta_img, avg_zeta_img, min_zeta_img, max_zeta_txt, avg_zeta_txt, min_zeta_txt, cur_eta_I, cur_eta_T