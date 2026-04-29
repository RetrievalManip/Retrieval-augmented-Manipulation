"""Loss functions used by the RAM BOP training script."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NOCSLoss(nn.Module):
    def __init__(self, corr_wt):
        super().__init__()
        self.threshold = 0.1
        self.corr_wt = corr_wt

    def forward(self, gt_nocs, pred_nocs, weight):
        diff = torch.abs(pred_nocs - gt_nocs)
        less = torch.pow(diff, 2) / (2.0 * self.threshold)
        higher = diff - self.threshold / 2.0
        corr_loss = torch.where(diff > self.threshold, higher, less)
        corr_loss = torch.sum(torch.mean(torch.sum(corr_loss, dim=2), dim=1) * weight)

        return corr_loss


class MatchLoss(nn.Module):
    def __init__(self, wt):
        super().__init__()
        self.wt = wt

    def forward(self, match_matrix):
        soft_assign = F.softmax(match_matrix, dim=2)
        log_assign = F.log_softmax(match_matrix, dim=2)
        entropy_loss = torch.mean(-torch.sum(soft_assign * log_assign, 2))
        entropy_loss = self.wt * entropy_loss

        return entropy_loss


class PoseNCE(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau

    def forward(self, query_feat, support_feat, support_error, positive_view_id):
        batch_size, _, feat_dim = support_feat.shape
        positive_choose = positive_view_id.unsqueeze(-1).repeat(1, 1, feat_dim)
        positive_feat = torch.gather(support_feat, 1, positive_choose).contiguous()

        l_pos = torch.exp(
            torch.bmm(query_feat.unsqueeze(1), positive_feat.permute(0, 2, 1)).squeeze(1) / self.tau
        )
        l_neg = torch.exp(
            torch.bmm(query_feat.unsqueeze(1), support_feat.permute(0, 2, 1)).squeeze(1) / self.tau
        ) * support_error

        logits_pos = torch.sum(l_pos, -1)
        logits_neg = torch.sum(l_neg, -1)
        loss = -torch.log(logits_pos / logits_neg)

        return loss.reshape(batch_size).mean()
