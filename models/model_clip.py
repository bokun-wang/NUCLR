from functools import partial

import timm
from transformers import AutoModel, RobertaModel

from .losses import InfoNCE_Loss, GCL_Loss, VICReg_Loss, RGCL_Loss, DGCL_Loss, DCL_Loss, CyCLIP_Loss, SigLipLoss, Spectral_Loss
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class CLIP(nn.Module):
    def __init__(self,
                 N_I,
                 N_T,
                 image_encoder,
                 text_encoder,
                 embed_dim,
                 init_model,
                 world_size,
                 ita_type,
                 gamma,
                 rho_init,
                 eta_init,
                 tau_init,
                 beta_u,
                 temp,
                 learnable_temp,
                 personalized_tau,
                 bsz,  # args.batch_size_train * args.world_size
                 vicreg_sim_coeff,
                 vicreg_std_coeff,
                 neg_zeta_init,
                 xi_init,
                 theta,
                 distributed,
                 start_epochs,
                 eta_I_ratio,
                 vanilla,
                 dcl_tau_plus,
                 cylambda_1,
                 cylambda_2,
                 siglip_t,
                 siglip_neg_bias,
                 siglip_bidir
                 ):
        super().__init__()

        self.temp = temp
        self.learnable_temp = learnable_temp
        self.personalized_tau = personalized_tau
        self.distributed = distributed

        if self.learnable_temp:
            if not personalized_tau:
                # learnable but not personalized temp
                self.temp = nn.Parameter(torch.ones([]) * self.temp)
            else:
                self.image_temp = nn.Parameter(torch.ones(N_I) * self.temp)
                self.text_temp = nn.Parameter(torch.ones(N_T) * self.temp)

        if ita_type == 'siglip':
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(siglip_t))
            self.logit_bias = nn.Parameter(torch.ones([]) * (-siglip_neg_bias))

        self.visual_encoder = timm.create_model(image_encoder, pretrained=init_model)
        self.visual_encoder.reset_classifier(0)

        if text_encoder == 'roberta-large':
            self.text_encoder = RobertaModel.from_pretrained(text_encoder)
            self.text_proj = nn.Linear(1024, embed_dim)
        else:
            self.text_encoder = AutoModel.from_pretrained(text_encoder)
            self.text_proj = nn.Linear(768, embed_dim)

        if not init_model:
            # execute when init_model == False
            # We will set init_model = True such that the following won't be executed
            self.text_encoder.init_weights()

        self.vision_proj = nn.Linear(self.visual_encoder.num_features, embed_dim)

        self.ita_type = ita_type

        if self.ita_type == 'infonce':
            if not personalized_tau:
                self.criterion = InfoNCE_Loss(world_size=world_size, personalized_tau=personalized_tau,
                                              temperature=self.temp)
            else:
                self.criterion = InfoNCE_Loss(world_size=world_size, personalized_tau=personalized_tau,
                                              image_tau=self.image_temp, text_tau=self.text_temp)

        elif self.ita_type == 'spectral':
            self.criterion = Spectral_Loss(world_size=world_size)

        elif self.ita_type == 'dcl':
            self.criterion = DCL_Loss(world_size=world_size, bsz=bsz, temperature=self.temp, tau_plus=dcl_tau_plus)
        
        elif self.ita_type == 'cyclip':
            self.criterion = CyCLIP_Loss(world_size=world_size, temperature=self.temp, cylambda_1=cylambda_1,
                                         cylambda_2=cylambda_2)

        elif self.ita_type == 'vicreg':
            self.criterion = VICReg_Loss(world_size=world_size, dim_size=embed_dim, sim_coeff=vicreg_sim_coeff,
                                         std_coeff=vicreg_std_coeff)
        elif self.ita_type == 'siglip':
            self.criterion = SigLipLoss(world_size=world_size, logit_scale=self.logit_scale, logit_bias=self.logit_bias, siglip_bidir=siglip_bidir)

        elif self.ita_type == 'gcl':
            self.criterion = GCL_Loss(N_I=N_I, N_T=N_T, world_size=world_size, gamma=gamma,
                                      temperature=self.temp, bsz=bsz)
        elif self.ita_type == 'rgcl':
            self.criterion = RGCL_Loss(N_I=N_I, N_T=N_T, world_size=world_size, gamma=gamma,
                                       rho_I=rho_init,
                                       rho_T=rho_init, tau_init=tau_init, bsz=bsz, eta_init=eta_init,
                                       beta_u=beta_u)
        elif self.ita_type == 'dgcl':
            self.criterion = DGCL_Loss(N_I=N_I, N_T=N_T, gamma=gamma, temperature=self.temp, world_size=world_size,
                                       bsz=bsz, neg_zeta_init=neg_zeta_init, xi_init=xi_init,
                                       theta=theta, start_epochs=start_epochs, eta_init=eta_init,
                                       eta_I_ratio=eta_I_ratio, vanilla=vanilla)
        else:
            raise NotImplementedError

    def forward(self, image, text, idx, text_idx, epoch, max_epoch):
        if self.learnable_temp:
            with torch.no_grad():
                if not self.personalized_tau:
                    self.temp.clamp_(0.001, 0.5)
                else:
                    self.image_temp.clamp_(0.001, 0.5)
                    self.text_temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder(image)
        image_embeds = self.vision_proj(image_embeds)
        image_feat = F.normalize(image_embeds, dim=-1)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, output_hidden_states=False)
        text_embeds = self.text_proj(text_output.last_hidden_state[:, 0, :])
        text_feat = F.normalize(text_embeds, dim=-1)

        info_dict = {}

        if self.ita_type in ['infonce', 'dcl', 'cyclip']:
            if self.personalized_tau:
                if self.distributed:
                    image_ids = concat_all_gather(idx)
                    text_ids = concat_all_gather(text_idx)
                else:
                    image_ids, text_ids = idx, text_idx
                loss_ita = self.criterion(image_feat, text_feat, image_ids, text_ids)
                info_dict['avg_image_tau'] = self.criterion.image_tau[image_ids].mean()
                info_dict['avg_text_tau'] = self.criterion.text_tau[text_ids].mean()

            else:
                loss_ita = self.criterion(image_feat, text_feat)
                if not self.learnable_temp:
                    avg_tau = torch.tensor(self.temp)
                else:
                    avg_tau = self.temp
                info_dict['avg_image_tau'] = avg_tau
                info_dict['avg_text_tau'] = avg_tau

        elif self.ita_type == 'vicreg':
            loss_ita = self.criterion(image_embeds, text_embeds)
            info_dict['avg_image_tau'] = 0.0
            info_dict['avg_text_tau'] = 0.0
        
        elif self.ita_type == 'siglip':
            loss_ita = self.criterion(image_embeds, text_embeds)
            info_dict['logit_scale'] = self.criterion.logit_scale.detach().clone().item()
            info_dict['logit_bias'] = self.criterion.logit_bias.detach().clone().item()
        
        elif self.ita_type == 'spectral':
            loss_ita = self.criterion(image_embeds, text_embeds)

        elif self.ita_type == 'gcl':
            if self.distributed:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
            else:
                image_ids, text_ids = idx, text_idx
            loss_ita = self.criterion(image_feat, text_feat, image_ids, text_ids, epoch)
            info_dict['avg_text_tau'] = 0.0
            info_dict['avg_image_tau'] = 0.0

        elif self.ita_type == 'rgcl':
            if self.distributed:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
            else:
                image_ids, text_ids = idx, text_idx
            loss_ita, avg_image_tau, avg_text_tau, cur_eta, grad_tau_image, grad_tau_text = self.criterion(
                image_feat, text_feat, image_ids, text_ids, epoch)
            info_dict = {'avg_image_tau': avg_image_tau, 'avg_text_tau': avg_text_tau, 'cur_eta': cur_eta,
                         'grad_tau_image': grad_tau_image, 'grad_tau_text': grad_tau_text}

        elif self.ita_type == 'dgcl':
            if self.distributed:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
            else:
                image_ids, text_ids = idx, text_idx
            loss_ita, max_zeta_img, avg_zeta_img, min_zeta_img, max_zeta_txt, avg_zeta_txt, min_zeta_txt, cur_eta_I, cur_eta_T = self.criterion(
                image_feat, text_feat,
                image_ids,
                text_ids,
                epoch,
                max_epoch)
            info_dict = {'max_zeta_img': max_zeta_img, 'avg_zeta_img': avg_zeta_img, 'min_zeta_img': min_zeta_img,
                         'max_zeta_txt': max_zeta_txt, 'avg_zeta_txt': avg_zeta_txt, 'min_zeta_txt': min_zeta_txt,
                         'cur_eta_I': cur_eta_I, 'cur_eta_T': cur_eta_T}
        else:
            raise NotImplementedError

        return loss_ita, info_dict


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
