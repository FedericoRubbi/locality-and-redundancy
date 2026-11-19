import torch.nn as nn
import torch.nn.functional as F

class TargetEmbeddings(nn.Module):
    def __init__(self, num_classes=100, embed_dim=384):
        super().__init__()
        self.embeddings = nn.Embedding(num_classes, embed_dim)

    def initialize_embeddings(self, image_embeddings, labels):
        for i in range(self.embeddings.num_embeddings):
            class_embeddings = image_embeddings[labels == i]
            if len(class_embeddings) > 0:
                centroid = class_embeddings.mean(dim=0)
                self.embeddings.weight.data[i] = F.normalize(centroid, p=2, dim=0)

    def forward(self, labels):
        return self.embeddings(labels)
