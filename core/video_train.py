import torch
import torch.nn.functional as F
from torch.cuda import amp
import torch.nn as nn

def reshape_video_frames(x1,seq_len=6):
    b, c, h, w = x1.size()
    x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
    return x1

def train_video_steps(iter,base,meter,scaler):


    for _ in range(base.steps):
        input1, input2, label1, label2 = iter.next_one()
        base.model_optimizer.zero_grad()
        rgb_imgs, rgb_pids = input1, label1
        ir_imgs, ir_pids = input2, label2
        rgb_imgs,  rgb_pids = rgb_imgs.to(base.device),rgb_pids.to(base.device).long()
        ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()
        rgb_imgs = reshape_video_frames(rgb_imgs,base.sequence_length)
        ir_imgs = reshape_video_frames(ir_imgs,base.sequence_length)
        modalities = torch.cat([
            torch.zeros_like(rgb_pids, dtype=torch.long),
            torch.ones_like(ir_pids,  dtype=torch.long)
        ], dim=0).to(base.device)

        with amp.autocast(enabled=True):
            features, cls_score = base.model(x1=rgb_imgs, x2=ir_imgs)

            pids = torch.cat([rgb_pids, ir_pids], dim=0)


            ide_loss = base.id_criterion(cls_score[0], pids)
            ide_loss_proj = base.id_criterion(cls_score[1], pids)


            # HHSA supervision: identity and triplet losses on the pooled DGF features.
            ide_loss_final = base.id_criterion(cls_score[2], pids)
            triplet_loss_final = base.triplet_criterion(features[3].squeeze(), pids)

            # GMA = HPA (prototype classification + center pulling) + HMA (modality alignment).
            ide_loss_hpc = base.id_criterion(cls_score[3], pids)


            hyp_align_loss = hyperbolic_prototype_alignment_loss(
                manifold=base.model.manifold,
                x_hyp=features[4].squeeze(),
                labels=pids,
                modalities=modalities,
                prototypes_hyp=base.model.hyp_classifier_TP.prototypes_hyp(),
                c=1.0,
                w_proto=0.05,
                w_cm=0.05,
            )

            triplet_loss_last = base.triplet_criterion(features[0].squeeze(), pids)
            triplet_loss = base.triplet_criterion(features[1].squeeze(), pids)
            triplet_loss_proj = base.triplet_criterion(features[2].squeeze(), pids)

            total_loss = ide_loss + ide_loss_proj + triplet_loss_last + triplet_loss + triplet_loss_proj + ide_loss_final + triplet_loss_final + 1.0 * ide_loss_hpc + 1.0 * hyp_align_loss
        scaler.scale(total_loss).backward()
        scaler.step(base.model_optimizer)
        scaler.update()


        meter.update({'pid_loss': ide_loss.data,
                      'pid_loss_proj': ide_loss_proj.data,
                      'pid_loss_final': ide_loss_final.data,
                      'pid_loss_hpc': ide_loss_hpc.data,
                      'tri_loss': triplet_loss.data,
                      'tri_loss_proj': triplet_loss_proj.data,
                      'tri_loss_last': triplet_loss_last.data,
                      'tri_loss_final': triplet_loss_final.data,

                      'hyp_align_loss': hyp_align_loss.data,

                      })
    return meter


@torch.no_grad()
def _group_indices_by_label(labels: torch.Tensor):

    uniq = labels.unique()
    groups = {}
    for y in uniq.tolist():
        groups[int(y)] = (labels == y).nonzero(as_tuple=False).squeeze(1)
    return groups

def hyperbolic_prototype_alignment_loss(
    manifold,
    x_hyp: torch.Tensor,
    labels: torch.Tensor,
    modalities: torch.Tensor,
    prototypes_hyp: torch.Tensor,
    c: float = 1.0,
    w_proto: float = 1.0,
    w_cm: float = 1.0,
):
    """
    Returns:
      loss_proto: pull both modality means to the shared prototype
      loss_cm:    pull modality means together (cross-modal)
    """
    x_hyp = manifold.proj(x_hyp, c=c)


    x_tan = manifold.logmap0(x_hyp, c=c)
    x_tan = manifold.proj_tan0(x_tan, c=c)

    groups = _group_indices_by_label(labels)

    loss_proto = x_hyp.new_zeros([])
    loss_cm = x_hyp.new_zeros([])
    cnt_proto = 0
    cnt_cm = 0

    for y, idx in groups.items():
        idx_vis = idx[(modalities[idx] == 0)]
        idx_ir  = idx[(modalities[idx] == 1)]


        mu_vis = None
        mu_ir = None

        if idx_vis.numel() > 0:
            mu_vis_tan = x_tan[idx_vis].mean(dim=0, keepdim=True)
            mu_vis = manifold.expmap0(mu_vis_tan, c=c)
            mu_vis = manifold.proj(mu_vis, c=c)

        if idx_ir.numel() > 0:
            mu_ir_tan = x_tan[idx_ir].mean(dim=0, keepdim=True)
            mu_ir = manifold.expmap0(mu_ir_tan, c=c)
            mu_ir = manifold.proj(mu_ir, c=c)

        p_y = prototypes_hyp[y].unsqueeze(0)


        # HPA center pulling: align both modality centers to the shared identity prototype.
        if mu_vis is not None:
            d2 = manifold.sqdist(mu_vis, p_y, c=c) if hasattr(manifold, "sqdist") else manifold.dist(mu_vis, p_y, c=c).pow(2)
            loss_proto = loss_proto + d2.mean()
            cnt_proto += 1
        if mu_ir is not None:
            d2 = manifold.sqdist(mu_ir, p_y, c=c) if hasattr(manifold, "sqdist") else manifold.dist(mu_ir, p_y, c=c).pow(2)
            loss_proto = loss_proto + d2.mean()
            cnt_proto += 1


        # HMA: align the visible and infrared centers of the same identity.
        if (mu_vis is not None) and (mu_ir is not None):
            d2 = manifold.sqdist(mu_vis, mu_ir, c=c) if hasattr(manifold, "sqdist") else manifold.dist(mu_vis, mu_ir, c=c).pow(2)
            loss_cm = loss_cm + d2.mean()
            cnt_cm += 1

    if cnt_proto > 0:
        loss_proto = loss_proto / cnt_proto
    if cnt_cm > 0:
        loss_cm = loss_cm / cnt_cm

    return w_proto * loss_proto + w_cm * loss_cm
