import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import os
from typing import Callable, Optional, Sequence, Tuple
from einops import rearrange
import math

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

class Attention_spatial(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)
        return x

class Backbone(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.0, local_token_num=8, global_token_num=8):
        super().__init__()
        clip_path = '..'
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained=os.path.join(clip_path, 'open_clip_pytorch_model.bin'))
        self.clip = self.clip.float()
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')

        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(1280,1024)
        self.text_fc = nn.Linear(1024,1024)


        self.chanel_score_text = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1, stride=1, bias=False),
                                    nn.Tanh(),
                                    nn.Conv1d(hidden_dim // 2, global_token_num, kernel_size=1, stride=1, bias=False),
                                    nn.Softmax(dim=-1))
        
        self.chanel_score = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1, stride=1, bias=False),
                                    nn.Tanh(),
                                    nn.Conv1d(hidden_dim // 2, global_token_num, kernel_size=1, stride=1, bias=False),
                                    nn.Softmax(dim=-1))

        self.cross_attention = Cross_attention(1280, n_head=4)
        self.cross_attention_text = Cross_attention_1d(1024, n_head=4)
        self.attention_spatial = Attention_spatial(1280)
        self.seg_fusion = Fusion_Embed(embed_dim=1280, bias=True)
        self.segText_fusion = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1, stride=1, bias=True)

        self.local_token_num = local_token_num
        self.global_token_num = global_token_num
        

    def visual_out(self, x):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.clip.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)

        x = self.clip.visual.patch_dropout(x)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x)
        pooled, tokens = self.clip.visual._global_pool(x)
        # print(tokens.shape)

        pooled = pooled @ self.clip.visual.proj

        
        return pooled, x
    
    def visual_seg_out(self, x, x_seg):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.clip.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)

        x = self.clip.visual.patch_dropout(x)
        x = self.clip.visual.ln_pre(x)

        x_seg = self.clip.visual.conv1(x_seg)  # shape = [*, width, grid, grid]
        x_seg = x_seg.reshape(x_seg.shape[0], x_seg.shape[1], -1)  # shape = [*, width, grid ** 2]
        x_seg = x_seg.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x_seg = torch.cat([_expand_token(self.clip.visual.class_embedding, x_seg.shape[0]).to(x_seg.dtype), x_seg], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x_seg = x_seg + self.clip.visual.positional_embedding.to(x_seg.dtype)

        x_seg = self.clip.visual.patch_dropout(x_seg)
        x_seg = self.clip.visual.ln_pre(x_seg)
        x_seg = x_seg.permute(1, 0, 2) 

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x) * self.clip.visual.transformer(x_seg) 
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.clip.visual.attn_pool is not None:
            if self.clip.visual.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.clip.visual.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.clip.visual.attn_pool(x)
                if self.clip.visual.attn_pool_type == 'parallel':
                    pooled = self.clip.visual.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.clip.visual.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.clip.visual.attn_pool(x)
                x = self.clip.visual.ln_post(x)
                pooled, tokens = self.clip.visual._global_pool(x)
        elif self.clip.visual.final_ln_after_pool:
            pooled, tokens = self.clip.visual._global_pool(x)
            pooled = self.clip.visual.ln_post(pooled)

        else:
            x = self.clip.visual.ln_post(x)
            pooled, tokens = self.clip.visual._global_pool(x)


        if self.clip.visual.proj is not None:
            pooled = pooled @ self.clip.visual.proj

        if self.clip.visual.output_tokens:
            return pooled, tokens
        
        return pooled, x
    
    def text_out(self, text):
        cast_dtype = self.clip.transformer.get_cast_dtype()

        x = self.clip.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        pooled, tokens = text_global_pool(x, text, self.clip.text_pool_type)
        if self.clip.text_projection is not None:
            if isinstance(self.clip.text_projection, nn.Linear):
                pooled = self.clip.text_projection(x)
            else:
                pooled = pooled @ self.clip.text_projection

        return pooled, x

    def extract_img_fea(self, x_ori):
        global_fea, local_fea = self.visual_out(x_ori)
        global_fea = global_fea.unsqueeze(1)
        x = self.fc(local_fea.float())
        global_tokens = torch.matmul(self.chanel_score(global_fea.transpose(1, 2)), global_fea)#self.global_attention(global_fea.unsqueeze(1))
        local_tokens = torch.matmul(self.chanel_score(x.transpose(1, 2)), x)
        return torch.cat([global_tokens, local_tokens], dim=1), (global_fea, global_fea.unsqueeze(1))

    def extract_img_seg_cross_fea(self, x_ori, x_seg):
        global_fea, local_fea = self.visual_out(x_ori)
        global_seg, local_seg = self.visual_out(x_seg)

        out_enc_level4_A, out_enc_level4_B = self.cross_attention(to_4d(local_fea[:, 1:, :], h=16, w=16), to_4d(local_seg[:, 1:, :], h=16, w=16))
        out_enc_level4 = self.seg_fusion(out_enc_level4_A, out_enc_level4_B)
        x = to_3d(out_enc_level4)#self.attention_spatial()

        global_fea = torch.cat([(x[:, 0, :] @ self.clip.visual.proj + global_fea).unsqueeze(1) , global_fea.unsqueeze(1), global_seg.unsqueeze(1)], dim=1)
        x = self.fc(x.float())
        global_tokens = torch.matmul(self.chanel_score(global_fea.transpose(1, 2)), global_fea)#self.global_attention(global_fea.unsqueeze(1))
        
        local_tokens = torch.matmul(self.chanel_score(x.transpose(1, 2)), x)
        return torch.cat([global_tokens, local_tokens], dim=1), (global_fea, global_seg.unsqueeze(1))
    

    def extract_text_fea_segGuide(self, txt, seg_local_fea):
        txt = self.tokenizer(txt).cuda()
        global_fea, x = self.text_out(txt)
        out_enc_level4_A, out_enc_level4_B = self.cross_attention_text(global_fea.unsqueeze(1).transpose(1, 2), seg_local_fea.transpose(1, 2))
        #print(out_enc_level4_A.shape, out_enc_level4_B.shape)
        seg_text = self.segText_fusion(torch.concat([out_enc_level4_A, out_enc_level4_B], dim=1)).transpose(1, 2)
        global_fea = torch.cat([seg_text + global_fea.unsqueeze(1), global_fea.unsqueeze(1), seg_local_fea], dim=1)
        x = self.text_fc(x.float())
        # global_tokens = self.global_attention(global_fea.unsqueeze(1))
        # local_tokens = self.local_attention(self.text_fc(x.float()))
        global_tokens = torch.matmul(self.chanel_score_text(global_fea.transpose(1, 2)), global_fea)#self.global_attention(global_fea.unsqueeze(1))
        local_tokens = torch.matmul(self.chanel_score_text(x.transpose(1, 2)), x)
        return torch.cat([global_tokens, local_tokens], dim=1), (global_fea, x)
    
    def extract_text_fea(self, txt):
        txt = self.tokenizer(txt).cuda()
        global_fea, x = self.text_out(txt)
        global_fea = global_fea.unsqueeze(1)
        x = self.text_fc(x.float())
        global_tokens = torch.matmul(self.chanel_score_text(global_fea.transpose(1, 2)), global_fea)#self.global_attention(global_fea.unsqueeze(1))
        local_tokens = torch.matmul(self.chanel_score_text(x.transpose(1, 2)), x)
        return torch.cat([global_tokens, local_tokens], dim=1), (global_fea, x)

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.cross_attention_text = Cross_attention_1d(512)

        self.MLP = nn.Sequential(nn.Conv1d(in_channels * 2, in_channels, kernel_size=1, stride=1, bias=True),
                                    nn.LeakyReLU(),
                                    nn.Conv1d(in_channels, out_channels * (1 + self.use_affine_level), kernel_size=1, stride=1, bias=True),
                                    nn.Sigmoid(),
                                )

    def forward(self, x, text_embed):
        text_embed_ = torch.cat([x, text_embed], dim=-1).transpose(1, 2)
        batch = x.shape[0]
        chanel = x.shape[1] * 2
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed_).transpose(1, 2).reshape(batch, chanel, -1).chunk(2, dim=1)
            # print(gamma.shape, beta.shape)
            x = gamma * x + (beta) * text_embed
        return x



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)



class Cross_attention_1d(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv1d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv1d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv1d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv1d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, length = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        # x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, length)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        # x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, length)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bncl, bncy -> bnly", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, length, -1)
        attn_A = torch.softmax(attn_A, -1)
        # attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnly, bncy -> bncl", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, length))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bncl, bncy -> bnly", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, length, -1)
        attn_B = torch.softmax(attn_B, -1)
        # attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnly, bncy -> bncl", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, length))
        out_B = out_B + x_B

        return out_A, out_B


class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B

class OFFSET(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.0, local_token_num=8, global_token_num=8, t=10):
        super().__init__()
        self.backbone = Backbone(hidden_dim, dropout, local_token_num, global_token_num)
        self.loss_T = nn.Parameter(torch.tensor([10.]))
        self.local_weight = nn.Parameter(torch.tensor([1.0 for _ in range(local_token_num * 1 + global_token_num * 1)]))

        self.t = t
        
        self.affine = FeatureWiseAffine(hidden_dim, hidden_dim, use_affine_level=True)

    def target_fea(self, tag, tag_seg):
        tag_token, (tag_global_fea, tag_local_fea) = self.backbone.extract_img_seg_cross_fea(tag, tag_seg)
        return tag_token#, ref_mask
    
    def compose_feature(self, ref, mod, ref_seg):
        ref_token, (ref_global_fea, ref_local_fea) = self.backbone.extract_img_seg_cross_fea(ref, ref_seg)
        mod_token, (global_text_fea, local_text_fea) = self.backbone.extract_text_fea_segGuide(mod, ref_local_fea)
        

        fuse_local = self.affine(ref_token, mod_token)

        return fuse_local

    def extract_retrieval_compose(self, ref, mod, ref_seg):
        fuse_local = self.compose_feature(ref, mod, ref_seg)
        fuse_local = F.normalize(torch.mean(fuse_local, dim=1), p=2, dim=-1)

        return fuse_local

    def extract_retrieval_target(self, tag, tag_seg):
        tag_local = self.target_fea(tag, tag_seg)
        tag_local = F.normalize(torch.mean(tag_local, dim=1), p=2, dim=-1)
        return tag_local

    def compute_loss(self, ref, mod, tag, ref_seg, tag_seg, cluster_result=None, index=None):
        fuse_local = self.compose_feature(ref, mod, ref_seg)
        tag_local = self.target_fea(tag, tag_seg)
        loss = {}

        retrieval_query = F.normalize(torch.mean(fuse_local, dim=1), p=2, dim=-1)
        retrieval_target = F.normalize(torch.mean(tag_local, dim=1), p=2, dim=-1)


        tag_feature = (F.normalize(tag_local, p=2, dim=-1) * self.local_weight.unsqueeze(0).unsqueeze(-1)).flatten(1)

        loss['rank'] = self.info_nce(retrieval_query, retrieval_target)
        loss['fr'] = self.kl_div(retrieval_query, retrieval_target, tag_feature, tag_feature, self.t)

        return loss
    
    def forward(self, img, img_seg, mod):
        return self.extract_retrieval_compose(img, mod, img_seg)

    
    def mask_constraint(self, mask1, mask2):
        mask = mask1 + mask2
        y = torch.ones_like(mask).float().cuda()
        return F.mse_loss(mask,y)

    def info_nce(self, query, target):
        x = torch.mm(query, target.T)
        labels = torch.arange(query.shape[0]).long().cuda()
        return F.cross_entropy(x * self.loss_T, labels)

    
    def kl_div(self, x1, y1, x2, y2, t):
        x1 = F.normalize(x1, p=2, dim=-1)
        y1 = F.normalize(y1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        y2 = F.normalize(y2, p=2, dim=-1)

        x1_y1 = torch.mm(x1, y1.T) / t
        x2_y2 = torch.mm(x2, y2.T) / t

        log_soft_x1 = F.log_softmax(x1_y1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x2_y2), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')

        return kl




