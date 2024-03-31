import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

def sim_stats(all_image_features, all_text_features, logits_per_image, logits_per_text, labels, num_logits):
    img_num = all_image_features.size(0)
    mask = torch.eye(img_num).cuda()
    img_to_img_sim = ((all_image_features @ all_image_features.T) * (1 - mask)).detach().sum() / (img_num * (img_num-1))
    txt_to_txt_sim = ((all_text_features @ all_text_features.T) * (1 - mask)).detach().sum() / (img_num * (img_num-1))
    img_to_img_nn_sim = ((all_image_features @ all_image_features.T) * (1 - mask)-2 * mask).detach().max(dim=1)[0].mean()
    txt_to_txt_nn_sim = ((all_text_features @ all_text_features.T) * (1 - mask)-2 * mask).detach().max(dim=1)[0].mean()
    img_to_pos_txt_sim = ((all_image_features @ all_text_features.T) * mask).detach().sum() / img_num
    img_to_neg_txt_sim = ((all_image_features @ all_text_features.T) * (1 - mask)).detach().sum() / (img_num * (img_num-1))
    img_to_hard_neg_txt_sim = (((all_image_features @ all_text_features.T) * (1 - mask)+ mask * (-2.)).max(dim=1))[0].mean()
    img_to_pos_minus_neg_txt_sim = img_to_pos_txt_sim - img_to_neg_txt_sim
    img_to_pos_minus_hard_neg_txt_sim = img_to_pos_txt_sim - img_to_hard_neg_txt_sim
    
    txt_to_pos_img_sim = ((all_text_features @ all_image_features.T) * mask).detach().sum() / img_num
    txt_to_neg_img_sim = ((all_text_features @ all_image_features.T) * (1 - mask)).detach().sum() / (img_num * (img_num-1))
    txt_to_hard_neg_img_sim = (((all_text_features @ all_image_features.T) * (1 - mask)+ mask * (-2.)).max(dim=1))[0].mean()
    txt_to_pos_minus_neg_img_sim = txt_to_pos_img_sim - txt_to_neg_img_sim
    txt_to_pos_minus_hard_neg_img_sim = txt_to_pos_img_sim - txt_to_hard_neg_img_sim
    
    targets = F.one_hot(labels, num_classes=logits_per_image.size(1)).float()
    prob_per_image = F.softmax(logits_per_image, 1)
    neg_prob_per_image = prob_per_image.clone()
    neg_prob_per_image[mask.bool()] = 0.
    hard_neg_indices = F.one_hot(neg_prob_per_image.max(dim=1)[1], num_classes=num_logits)
    img_anchor_grad_from_hard_neg = F.normalize((neg_prob_per_image * hard_neg_indices) @ all_text_features, dim=1)
    img_anchor_grad_from_neg = F.normalize(neg_prob_per_image @ all_text_features, dim=1)
    img_anchor_grad_from_pos = F.normalize((prob_per_image * mask - mask) @ all_text_features, dim=1)

    img_anchor_txt_pos_neg_grad_sim = (img_anchor_grad_from_pos * img_anchor_grad_from_neg).sum(dim=1).mean()
    img_anchor_txt_pos_hard_neg_grad_sim = (img_anchor_grad_from_pos * img_anchor_grad_from_hard_neg).sum(dim=1).mean()
    
    prob_per_txt = F.softmax(logits_per_text, 1)
    neg_prob_per_txt = prob_per_txt.clone()
    neg_prob_per_txt[mask.bool()] = 0.
    hard_neg_txt_indices = F.one_hot(neg_prob_per_txt.max(dim=1)[1], num_classes=num_logits)
    txt_anchor_grad_from_hard_neg = F.normalize((neg_prob_per_txt * hard_neg_txt_indices) @ all_image_features, dim=1)
    txt_anchor_grad_from_neg = F.normalize(neg_prob_per_txt @ all_image_features, dim=1)
    txt_anchor_grad_from_pos = F.normalize((prob_per_txt * mask - mask) @ all_image_features, dim=1)

    txt_anchor_img_pos_neg_grad_sim = (txt_anchor_grad_from_pos * txt_anchor_grad_from_neg).sum(dim=1).mean()
    txt_anchor_img_pos_hard_neg_grad_sim = (txt_anchor_grad_from_pos * txt_anchor_grad_from_hard_neg).sum(dim=1).mean()

    
    sims = [img_to_img_sim.item(), 
            txt_to_txt_sim.item(), 
            img_to_pos_txt_sim.item(), 
            img_to_neg_txt_sim.item(), 
            img_to_hard_neg_txt_sim.item(),
            img_to_pos_minus_neg_txt_sim.item(), 
            img_to_pos_minus_hard_neg_txt_sim.item(),
            txt_to_pos_img_sim.item(),
            txt_to_neg_img_sim.item(),
            txt_to_hard_neg_img_sim.item(),
            txt_to_pos_minus_neg_img_sim.item(),
            txt_to_pos_minus_hard_neg_img_sim.item(),
            img_anchor_txt_pos_neg_grad_sim.item(),
            img_anchor_txt_pos_hard_neg_grad_sim.item(),
            txt_anchor_img_pos_neg_grad_sim.item(),
            txt_anchor_img_pos_hard_neg_grad_sim.item(),
            img_to_img_nn_sim.item(),
            txt_to_txt_nn_sim.item(),]

    return sims


def get_grad(p, k, tau, targets):
    logits = p @ k.T / tau
    targets = F.one_hot(targets, num_classes=logits.size(1)).float()
    prob = F.softmax(logits, 1)
    grad_p = (prob - targets) @ k / tau / targets.size(0)
    embed_size = p.size(1)
    prob_targets_repeat = (prob - targets).t().repeat(1, embed_size).view(-1,embed_size, p.size(0))
    grad_k = (prob_targets_repeat * (p.t() / tau).unsqueeze(0)).sum(-1) / targets.size(0)

    return grad_p, grad_k
     
    
class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        return total_loss


    
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss

class KDClipLoss(nn.Module):

    def __init__(
            self,
            args,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.args = args

        if args.t_embed_dim != args.s_embed_dim:
            self.visual_proj = nn.Linear(args.s_embed_dim, args.t_embed_dim)
            self.text_proj = nn.Linear(args.s_embed_dim, args.t_embed_dim)
        
        if args.alpha_afd_loss > 0.:
            self.visual_fusion_proj = nn.Linear(args.s_embed_dim+args.t_embed_dim, args.s_embed_dim)
            self.text_fusion_proj = nn.Linear(args.s_embed_dim+args.t_embed_dim, args.s_embed_dim)
            
        # cache state
        self.prev_num_logits = 0
        self.kl_loss = DistillKL(T=1)
        self.cross_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fusion_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, \
        t_image_features, t_text_features, t_logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            t_all_image_features, t_all_text_features = gather_features(
                t_image_features, t_text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            t_logits_per_image = t_logit_scale * t_all_image_features @ t_all_text_features.T
            t_logits_per_text = t_logits_per_image.T

            normalized_image_features = F.normalize(image_features, dim=1)
            normalized_text_features = F.normalize(text_features, dim=1)
            normalized_all_image_features = F.normalize(all_image_features, dim=1)
            normalized_all_text_features = F.normalize(all_text_features, dim=1)
            
            if self.local_loss:
                logits_per_image = logit_scale * normalized_image_features @ normalized_all_text_features.T
                logits_per_text = logit_scale * normalized_text_features @ normalized_all_image_features.T
            else:
                logits_per_image = logit_scale * normalized_all_image_features @ normalized_all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * normalized_image_features @ normalized_text_features.T
            logits_per_text = logit_scale * normalized_text_features @ normalized_image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        if self.args.t_embed_dim != self.args.s_embed_dim:
            all_image_features = self.visual_proj(all_image_features)
            all_text_features = self.text_proj(all_text_features)
            
        normalized_all_image_features = F.normalize(all_image_features, dim=1)
        normalized_all_text_features = F.normalize(all_text_features, dim=1)
        fd_loss = F.mse_loss(normalized_all_image_features, t_all_image_features) +\
            F.mse_loss(normalized_all_text_features, t_all_text_features)
            
        logits_per_s_image_to_t_text = self.cross_logit_scale * normalized_all_image_features @ t_all_text_features.T
        logits_per_s_text_to_t_image = self.cross_logit_scale * normalized_all_text_features @ t_all_image_features.T
        
        task_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        ckd_loss = torch.tensor(0.).cuda() 
        icl_loss = torch.tensor(0.).cuda() 
        cross_kd_loss = torch.tensor(0.).cuda() 
        gd_loss = torch.tensor(0.).cuda() 
        afd_loss = torch.tensor(0.).cuda() 
        
        icl_loss = (
            F.cross_entropy(logits_per_s_image_to_t_text, labels) +
            F.cross_entropy(logits_per_s_text_to_t_image, labels)
            ) / 2
        
        ckd_loss = (self.kl_loss(logits_per_image, t_logits_per_image.detach()) +\
            self.kl_loss(logits_per_text, t_logits_per_text.detach())) / 2
        
        cross_kd_loss = (self.kl_loss(logits_per_s_image_to_t_text, t_logits_per_image.detach()) +\
            self.kl_loss(logits_per_s_text_to_t_image, t_logits_per_text.detach())) / 2
        #kd_loss = (F.cross_entropy(logits_per_image, F.softmax(, dim=1)) \
        #    + F.cross_entropy(logits_per_text, F.softmax(t_logits_per_text.detach(), dim=1))) / 2
        
        
        if self.args.alpha_gd_loss > 0.:
            with torch.no_grad():
                t_grad_p_img, t_grad_k_txt = get_grad(t_all_image_features, t_all_text_features, t_logit_scale, labels)
                t_grad_p_txt, t_grad_k_img = get_grad(t_all_text_features, t_all_image_features, t_logit_scale, labels)
            
            s_grad_p_img, s_grad_k_txt = get_grad(normalized_all_image_features, normalized_all_text_features, logit_scale, labels)
            s_grad_p_txt, s_grad_k_img = get_grad(normalized_all_text_features, normalized_all_image_features, logit_scale, labels)

            gd_loss = F.mse_loss(s_grad_p_img, t_grad_p_img.detach()) +\
                F.mse_loss(s_grad_k_txt, t_grad_k_txt.detach()) +\
                    F.mse_loss(s_grad_p_txt, t_grad_p_txt.detach()) +\
                        F.mse_loss(s_grad_k_img, t_grad_k_img.detach()) 
        
        if self.args.alpha_afd_loss > 0.:
            img_fusion_feat = torch.cat([normalized_all_image_features, t_all_image_features], dim=1)
            txt_fusion_feat = torch.cat([normalized_all_text_features, t_all_text_features], dim=1)
            img_fusion_feat = self.visual_fusion_proj(img_fusion_feat)
            txt_fusion_feat = self.text_fusion_proj(txt_fusion_feat)
            img_fusion_feat = F.normalize(img_fusion_feat, dim=1)
            txt_fusion_feat = F.normalize(txt_fusion_feat, dim=1)
            
            logits_per_fusion_image = self.fusion_logit_scale * img_fusion_feat @ txt_fusion_feat.T
            logits_per_fusion_text = logits_per_fusion_image.T
            afd_loss = (
                F.cross_entropy(logits_per_fusion_image, labels) +
                F.cross_entropy(logits_per_fusion_text, labels)
            ) / 2
            
            
        ckd_loss = self.args.alpha_ckd_loss * ckd_loss
        icl_loss = self.args.alpha_icl_loss * icl_loss
        cross_kd_loss = self.args.alpha_cross_kd_loss * cross_kd_loss
        fd_loss = self.args.alpha_fd_loss * fd_loss
        gd_loss = self.args.alpha_gd_loss * gd_loss
        afd_loss = self.args.alpha_afd_loss * afd_loss
        
        return task_loss, ckd_loss, icl_loss, cross_kd_loss, fd_loss, gd_loss, afd_loss