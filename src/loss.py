import torch.nn.functional as F
from utils import margin_schedule
import config
def cosine_sim_loss(anchor_embs, positive_embs):
    return 1 - F.cosine_similarity(anchor_embs, positive_embs).mean()
  
def semi_hard_triplet_loss(anchor_embs, positive_embs, negative_embs, margin=config.MARGIN):
    
    d_ap = F.pairwise_distance(anchor_embs, positive_embs)  
    d_an = F.pairwise_distance(anchor_embs, negative_embs)  

    semi_hard_mask = (d_an > d_ap) & (d_an < (d_ap + margin))

    if not semi_hard_mask.any():
        losses = F.relu(d_ap - d_an + margin)
        return losses.mean()

    losses = F.relu(d_ap[semi_hard_mask] - d_an[semi_hard_mask] + margin)
    return losses.mean()

def hard_triplet_loss(anchor_embs, positive_embs, negative_embs, margin=config.MARGIN):
    d_ap = F.pairwise_distance(anchor_embs, positive_embs)
    d_an = F.pairwise_distance(anchor_embs, negative_embs)
    
    hard_mask = d_an < d_ap
    
    
    if not hard_mask.any():
        losses = F.relu(d_ap - d_an + margin)
        return losses.mean()
    
    losses = F.relu(d_ap[hard_mask] - d_an[hard_mask] + margin)
    return losses.mean()
  
def hybrid_triplet_loss(anchor_embs, positive_embs, negative_embs, epoch, margin_scheduling = True, margin = config.MARGIN, alpha = config.ALPHA):
  d_an = F.pairwise_distance(anchor_embs,negative_embs)
  d_ap = F.pairwise_distance(anchor_embs,positive_embs)
  if margin_scheduling:
    margin = margin_schedule(epoch)

  hard_mask = d_an < d_ap

  semi_hard_mask = (d_an > d_ap) & (d_an < (d_ap + margin))
  
  hard_loss = None
  semi_loss = None
  if hard_mask.any():
    hard_loss = F.relu(d_ap[hard_mask] - d_an[hard_mask] + margin)
    hard_loss = hard_loss.mean()
    
  if semi_hard_mask.any():
    semi_loss = F.relu(d_ap[semi_hard_mask] - d_an[semi_hard_mask] + margin)  
    semi_loss = semi_loss.mean()
    
  base_loss = 0.0
  if hard_loss is not None and semi_loss is not None:
    base_loss = alpha * hard_loss + (1 - alpha) * semi_loss
    
  elif hard_loss is not None:
    base_loss = hard_loss
    
  elif semi_loss is not None:
    base_loss = semi_loss

  else:
    base_loss = F.relu(d_ap - d_an + margin).mean()
  
  cosine_weight = 0.25 if epoch > 10 else 0.0
  return base_loss + cosine_weight * cosine_sim_loss(anchor_embs,positive_embs)