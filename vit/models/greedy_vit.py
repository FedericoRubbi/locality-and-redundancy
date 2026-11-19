import torch
from torch import nn

class BlockAuxiliaryHead(nn.Module):
    def __init__(self, dim, num_classes, pool='cls'):
        super().__init__()
        self.pool = pool
        self.norm = nn.LayerNorm(dim)
        self.to_latent = nn.Identity()
        self.linear = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        if self.pool == 'cls':
            pooled = x[:, 0]
        else:
            pooled = x.mean(dim=1)
            
        pooled = self.norm(pooled)
        pooled = self.to_latent(pooled)
        return self.linear(pooled)

class GreedyTransformerBlock(nn.Module):
    def __init__(self, vit_layer, dim, num_classes, pool='cls', is_first=False):
        super().__init__()
        self.attn = vit_layer[0]
        self.ff = vit_layer[1]
        self.aux_head = BlockAuxiliaryHead(dim, num_classes, pool)
        self.is_first = is_first

    def forward(self, x):
        if not self.is_first:
            x = x.detach()
            x.requires_grad_(True)
        
        h = self.attn(x) + x
        out = self.ff(h) + h
        
        logits = self.aux_head(out)
        
        return out, logits

class GreedyViT(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.image_size = vit_model.pos_embedding.shape[1]
        self.pool = vit_model.pool
        self.num_classes = vit_model.mlp_head.out_features if vit_model.mlp_head else 0
        dim = vit_model.pos_embedding.shape[-1]
        
        self.to_patch_embedding = vit_model.to_patch_embedding
        self.cls_token = vit_model.cls_token
        self.pos_embedding = vit_model.pos_embedding
        self.dropout = vit_model.dropout
        
        self.layers = nn.ModuleList()
        for idx, layer in enumerate(vit_model.transformer.layers):
            is_first = (idx == 0)
            greedy_block = GreedyTransformerBlock(layer, dim, self.num_classes, self.pool, is_first=is_first)
            self.layers.append(greedy_block)
            
        self.norm = vit_model.transformer.norm
        
    def forward(self, img):
        batch = img.shape[0]
        x = self.to_patch_embedding(img)

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        seq = x.shape[1]
        x = x + self.pos_embedding[:seq]
        x = self.dropout(x)

        block_outputs = []
        block_logits = []
        
        for block in self.layers:
            x, logits = block(x)
            block_outputs.append(x)
            block_logits.append(logits)
            
        x = self.norm(x)
        
        return block_outputs, block_logits
