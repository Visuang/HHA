from __future__ import print_function, absolute_import
import numpy as np
import torch

def evaluate_vcm(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):

    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)


    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):

        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]


        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)


        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):

            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.


        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def evaluate_bupt(distmat, query_pids, gallery_pids, opt):
    if isinstance(distmat, torch.Tensor):
        distmat = distmat.detach().cpu().numpy()
    if isinstance(query_pids, torch.Tensor):
        query_pids = query_pids.detach().cpu().numpy()
    if isinstance(gallery_pids, torch.Tensor):
        gallery_pids = gallery_pids.detach().cpu().numpy()

    num_q, num_g = distmat.shape
    assert num_q == len(query_pids) and num_g == len(gallery_pids)

    max_rank = min(opt.max_rank, num_g)

    indices = np.argsort(distmat, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, np.newaxis]).astype(np.int32)

    num_valid_query = 0
    all_cmc, all_ap = [], []
    for qi in range(num_q):
        orig_cmc = matches[qi]


        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_query += 1.


        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc =  np.asarray(tmp_cmc) * orig_cmc
        ap = tmp_cmc.sum() / num_rel
        all_ap.append(ap)

    assert num_valid_query > 0, "No query appears in gallery."

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_query
    mAP = np.mean(all_ap)

    return all_cmc, mAP
