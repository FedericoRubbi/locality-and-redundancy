import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch.vit import Attention, FeedForward

class BlockAuxiliaryHead(nn.Module):
    def __init__(self, dim, hidden_dim, num_classes, pool):
        super().__init__()
        self.pool = pool
        self.norm = nn.LayerNorm(dim)
        self.to_latent = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
        )
        self.linear = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        if self.pool == 'cls':
            pooled = x[:, 0]
        elif self.pool == 'mean':
            pooled = x.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pool type: {self.pool}")
            
        pooled = self.norm(pooled)
        pooled = self.to_latent(pooled)
        return self.linear(pooled)

class SharedTransposeMLP(nn.Module):
    def __init__(self, encoder_ff: FeedForward, dim, hidden_dim):
        super().__init__()
        self.encoder_ff = encoder_ff
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.act = nn.GELU()
        
    def forward(self, x):
        
        W_1 = self.encoder_ff.net[4].weight
        x = self.norm1(x)
        x = F.linear(x, W_1.t())
        
        x = self.act(x)
        
        W_0 = self.encoder_ff.net[1].weight
        x = self.norm2(x)
        x = F.linear(x, W_0.t())
        
        return x

class InfoViTV1Block(nn.Module):
    def __init__(self, vit_layer, dim, num_classes, pool='mean', is_first=False, heads=8, dim_head=64, mlp_dim=768):
        super().__init__()
        self.dim = dim
        self.is_first = is_first
        
        self.inter_norm = nn.LayerNorm(dim) if not is_first else nn.Identity()
        
        self.attn = vit_layer[0]
        self.ff = vit_layer[1]
        
        self.aux_head = BlockAuxiliaryHead(dim, mlp_dim, num_classes, pool)
        
        self.dec_attn = Attention(dim, heads=heads, dim_head=dim_head)
        
        if hasattr(self.ff, 'net') and not hasattr(self.ff, 'fn'):
            encoder_ff = self.ff
        elif hasattr(self.ff, 'fn'):
            encoder_ff = self.ff.fn
        else:
            raise ValueError("Could not locate FeedForward in vit_layer")
            
        nn.init.orthogonal_(encoder_ff.net[1].weight)
        nn.init.orthogonal_(encoder_ff.net[4].weight)
            
        self.dec_mlp = SharedTransposeMLP(encoder_ff, dim=dim, hidden_dim=mlp_dim)
        
    def get_orthogonality(self):
        if hasattr(self.ff, 'net') and not hasattr(self.ff, 'fn'):
            encoder_ff = self.ff
        elif hasattr(self.ff, 'fn'):
            encoder_ff = self.ff.fn
            
        W0 = encoder_ff.net[1].weight
        W1 = encoder_ff.net[4].weight
        
        I_dim = torch.eye(W0.size(1), device=W0.device)
        dev0 = torch.norm(W0.t() @ W0 - I_dim, p=2)
        
        dev1 = torch.norm(W1 @ W1.t() - I_dim, p=2)
        
        return (dev0.item() + dev1.item()) / 2.0
        
    def forward(self, x):
        if not self.is_first:
            x = x.detach()
            x.requires_grad_(True)
            
        x_norm = self.inter_norm(x)
        target = x_norm
        
        h = self.attn(x_norm) + x_norm
        out = self.ff(h) + h
        
        logits = self.aux_head(out)
        
        dec_h = self.dec_attn(out) + out
        reconstruction = self.dec_mlp(dec_h) + dec_h
        
        sim = F.cosine_similarity(reconstruction, target, dim=-1)
        rec_loss = (1.0 - sim).mean()
        
        return out, logits, rec_loss

class InfoViTV1(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.image_size = vit_model.pos_embedding.shape[1] 
        self.pool = vit_model.pool
        self.num_classes = vit_model.mlp_head[-1].out_features if hasattr(vit_model, 'mlp_head') and isinstance(vit_model.mlp_head, nn.Sequential) else vit_model.mlp_head.out_features if vit_model.mlp_head else 0
        dim = vit_model.pos_embedding.shape[-1]
        
        self.to_patch_embedding = vit_model.to_patch_embedding
        self.cls_token = vit_model.cls_token
        self.pos_embedding = vit_model.pos_embedding
        self.dropout = vit_model.dropout
        
        mlp_dim = vit_model.transformer.layers[0][1].fn.net[1].out_features if hasattr(vit_model.transformer.layers[0][1], 'fn') else vit_model.transformer.layers[0][1].net[1].out_features
        heads = vit_model.transformer.layers[0][0].fn.heads if hasattr(vit_model.transformer.layers[0][0], 'fn') else vit_model.transformer.layers[0][0].heads
        dim_head = vit_model.transformer.layers[0][0].fn.scale ** -2 if hasattr(vit_model.transformer.layers[0][0], 'fn') else 64
        dim_head = int(round(dim_head))
        
        self.layers = nn.ModuleList()
        for idx, layer in enumerate(vit_model.transformer.layers):
            is_first = (idx == 0)
            block = InfoViTV1Block(
                layer, dim, self.num_classes, self.pool, 
                is_first=is_first, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim
            )
            self.layers.append(block)
            
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
        block_rec_losses = []
        
        for block in self.layers:
            x, logits, rec_loss = block(x)
            block_outputs.append(x)
            block_logits.append(logits)
            block_rec_losses.append(rec_loss)
            
        x = self.norm(x)
        
        return block_outputs, block_logits, block_rec_losses
        
    def get_orthogonality(self):
        return [block.get_orthogonality() for block in self.layers]

def test_forward():
    from vit_pytorch import ViT
    vit = ViT(image_size=224, patch_size=16, num_classes=1000, dim=256, depth=3, heads=8, mlp_dim=512)
    model = InfoViTV1(vit)
    img = torch.randn(2, 3, 224, 224)
    out, logits, rec_losses = model(img)
    print("Pass: out length", len(out), "logits shape", logits[0].shape, "rec_losses", len(rec_losses))
    
    loss = (sum(rec_losses) + sum(l.mean() for l in logits))
    loss.backward()
    print("Pass: Backwards completed without errors.")
