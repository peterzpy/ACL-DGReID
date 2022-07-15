import torch
import torch.nn.functional as F


def intraCluster(feature, domain_ids, margin=0.1):
    feature = F.normalize(feature, 2, 1)
    loss = 0
    count = 0
    for i in range(3):
        if (domain_ids == i).sum().item() == 0:
            continue
        count += 1
        loss += (F.relu(torch.norm(feature[domain_ids==i] - feature[domain_ids==i].mean(), 2, 1) - margin) ** 2).mean()

    return loss / count


def interCluster(feature, domain_ids, margin=0.3):
    feature = F.normalize(feature, 2, 1)
    candidate_list = []
    loss = 0
    count = 0
    for i in range(3):
        if (domain_ids == i).sum().item() == 0:
            continue
        candidate_list.append(feature[domain_ids==i].mean())
    for i in range(len(candidate_list)):
        for j in range(i+1, len(candidate_list)):
            count += 1
            loss += F.relu(margin - torch.norm(candidate_list[i]-candidate_list[j], 2, 0)) ** 2
    
    return loss / count if count else domain_ids.float().mean()