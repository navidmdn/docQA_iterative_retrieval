import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np


def mhop_loss(model, batch, return_outputs=False):
    outputs = model(batch)

    loss_fct = CrossEntropyLoss(ignore_index=-1)

    all_ctx = torch.cat([outputs['c1'], outputs['c2']], dim=0) # B x h
    neg_ctx = torch.cat([outputs["neg_1"].unsqueeze(1), outputs["neg_2"].unsqueeze(1)], dim=1)  # B x 2 x h

    scores_1_hop = torch.mm(outputs["q"], all_ctx.t()) # B x 2B
    neg_scores_1 = torch.bmm(outputs["q"].unsqueeze(1), neg_ctx.transpose(1, 2)).squeeze(1) # B x 2
    scores_2_hop = torch.mm(outputs["q_sp1"], all_ctx.t()) # B x 2B
    neg_scores_2 = torch.bmm(outputs["q_sp1"].unsqueeze(1), neg_ctx.transpose(1, 2)).squeeze(1) # B x 2

    # mask the 1st hop
    bsize = outputs["q"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(outputs["q"].device)
    scores_1_hop = scores_1_hop.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1_hop)
    scores_1_hop = torch.cat([scores_1_hop, neg_scores_1], dim=1) # B x 2B+2
    scores_2_hop = torch.cat([scores_2_hop, neg_scores_2], dim=1) # B x 2B+2

    # if args.momentum:
    #     queue_neg_scores_1 = torch.mm(outputs["q"], model.module.queue.clone().detach().t())
    #     queue_neg_scores_2 = torch.mm(outputs["q_sp1"], model.module.queue.clone().detach().t())
    #
    #     # queue_neg_scores_1 = queue_neg_scores_1 / args.temperature
    #     # queue_neg_scores_2 = queue_neg_scores_2 / args.temperature
    #
    #     scores_1_hop = torch.cat([scores_1_hop, queue_neg_scores_1], dim=1)
    #     scores_2_hop = torch.cat([scores_2_hop, queue_neg_scores_2], dim=1)
    #     model.module.dequeue_and_enqueue(all_ctx.detach())
    #     # model.module.momentum_update_key_encoder()

    target_1_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device)
    target_2_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device) + outputs["q"].size(0)

    retrieve_loss = loss_fct(scores_1_hop, target_1_hop) + loss_fct(scores_2_hop, target_2_hop)

    return retrieve_loss if return_outputs is False else (retrieve_loss, outputs)


def mhop_eval(outputs):
    #todo: fix duplicate code
    all_ctx = torch.cat([outputs['c1'], outputs['c2']], dim=0)
    neg_ctx = torch.cat([outputs["neg_1"].unsqueeze(1), outputs["neg_2"].unsqueeze(1)], dim=1)

    scores_1_hop = torch.mm(outputs["q"], all_ctx.t())
    neg_scores_1 = torch.bmm(outputs["q"].unsqueeze(1), neg_ctx.transpose(1, 2)).squeeze(1)
    scores_2_hop = torch.mm(outputs["q_sp1"], all_ctx.t())
    neg_scores_2 = torch.bmm(outputs["q_sp1"].unsqueeze(1), neg_ctx.transpose(1, 2)).squeeze(1)

    bsize = outputs["q"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(outputs["q"].device)

    # making sure question score for c2 is -inf
    scores_1_hop = scores_1_hop.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1_hop)

    scores_1_hop = torch.cat([scores_1_hop, neg_scores_1], dim=1)
    scores_2_hop = torch.cat([scores_2_hop, neg_scores_2], dim=1)
    target_1_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device)
    target_2_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device) + outputs["q"].size(0)

    ranked_1_hop = scores_1_hop.argsort(dim=1, descending=True)
    ranked_2_hop = scores_2_hop.argsort(dim=1, descending=True)
    idx2ranked_1 = ranked_1_hop.argsort(dim=1)
    idx2ranked_2 = ranked_2_hop.argsort(dim=1)
    rrs_1, rrs_2 = [], []
    for t, idx2ranked in zip(target_1_hop, idx2ranked_1):
        # starts from 0 so +1
        rrs_1.append(1 / (idx2ranked[t].item() + 1))
    for t, idx2ranked in zip(target_2_hop, idx2ranked_2):
        rrs_2.append(1 / (idx2ranked[t].item() + 1))

    return {"rrs_1": np.mean(rrs_1), "rrs_2": np.mean(rrs_2), "rss_avg": np.mean([np.mean(rrs_1), np.mean(rrs_2)])}
