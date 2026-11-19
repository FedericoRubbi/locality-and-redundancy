import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_infonce(features, temperature=0.1):
    features = F.normalize(features, p=2, dim=1)
    
    similarity_matrix = torch.matmul(features, features.T)
    
    labels = torch.arange(features.shape[0], device=features.device)
    
    loss = F.cross_entropy(similarity_matrix / temperature, labels)
    
    return loss

def loss_supcon(features, labels, temperature=0.1):
    features = F.normalize(features, p=2, dim=1)
    
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    
    similarity_matrix = torch.matmul(features, features.T)
    
    loss = F.cross_entropy(similarity_matrix / temperature, mask)
    
    return loss

def loss_infomax(y_t, y_t_minus_1, temperature=0.1):
    y_t = F.normalize(y_t, p=2, dim=1)
    y_t_minus_1 = F.normalize(y_t_minus_1, p=2, dim=1)
    
    similarity_matrix = torch.matmul(y_t, y_t_minus_1.T)
    
    labels = torch.arange(y_t.shape[0], device=y_t.device)
    
    loss = F.cross_entropy(similarity_matrix / temperature, labels)
    
    return loss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        
        z = torch.cat((z_i, z_j), dim=0)
        
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        labels = torch.cat((torch.arange(batch_size) + batch_size, torch.arange(batch_size)), dim=0).to(sim_matrix.device)
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(sim_matrix.device)
        sim_matrix.masked_fill_(mask, -float('inf'))
        
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
