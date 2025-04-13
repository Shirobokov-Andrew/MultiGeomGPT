import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lmath import project, distance
from config.model_config import MultiGeomGPTConfig


class Rotary(nn.Module):
    """Rotary Positional Embeddings"""

    def __init__(self, dim: int, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            # emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = freqs.cos()[None, :, None, :].bfloat16()
            self.sin_cached = freqs.sin()[None, :, None, :].bfloat16()
        return self.cos_cached, self.sin_cached


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    rotated = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
    return rotated.type_as(x)


def custom_attention(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        dropout: nn.Dropout = None,
        curvature: float = None, 
        mode: str = 'euc', 
        scale: float = None, 
        ) -> torch.Tensor:
    if mode == "euc":
        return F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scale)
    elif mode == "sph":
        q_norm = F.normalize(query, p=2, dim=-1)
        k_norm = F.normalize(key, p=2, dim=-1)
        return F.scaled_dot_product_attention(q_norm, k_norm, value, is_causal=True, scale=scale)
    elif mode == "hyp":
        L, S = query.size(-2), key.size(-2)
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1)
        attn_bias.masked_fill_(mask, float('-inf'))

        lq = project(query, k=curvature).unsqueeze(-2)
        lk = project(key, k=curvature).unsqueeze(-3)
        dis = distance(lq, lk, k=curvature)
        attn_weight = 1 / (1e-6 + dis)

        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        if dropout is not None:
            attn_weight = dropout(attn_weight)
        return attn_weight @ value


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return x / (norm + self.eps) * self.weight


class MultiGeometryAttention(nn.Module):
    def __init__(self, config: MultiGeomGPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n1 = config.n1  # number of Euclidean heads
        self.n2 = config.n2  # number of Hyperbolic heads
        self.n3 = config.n3  # number of Spherical heads
        self.head_dim = config.n_embd // config.n_heads
        # self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(p=config.dropout)
        self.resid_dropout = nn.Dropout(p=config.dropout)
        assert self.n1 + self.n2 + self.n3 == self.n_heads

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.rotary = Rotary(self.head_dim)

        if self.n2 > 0:
            if config.attn_k_lr:
                # self.hyp_curvature = nn.Parameter(
                #     torch.full((1, self.n2, 1, 1), config.curvature))
                x = torch.randn(1, self.n2, 1, 1, device=self.qkv.weight.device)
                init_k = torch.exp(x) * config.curvature
                self.hyp_curvature = nn.Parameter(init_k)
            else:
                self.register_buffer('hyp_curvature',
                                     torch.full((1, self.n2, 1, 1), config.curvature))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).split(C, dim=2)
        q, k, v = [x.view(B, T, self.n_heads, self.head_dim) for x in qkv]

        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        split_sizes = [self.n1, self.n2, self.n3]
        qs = torch.split(q, split_sizes, dim=2)
        ks = torch.split(k, split_sizes, dim=2)
        vs = torch.split(v, split_sizes, dim=2)

        outputs = []
        if self.n1 > 0:
            attn = custom_attention(
                qs[0].transpose(1, 2), 
                ks[0].transpose(1, 2), 
                vs[0].transpose(1, 2), 
                mode='euc', 
                dropout=self.attn_dropout, 
                scale=1 / math.sqrt(self.head_dim),
                )
            outputs.append(attn.transpose(1, 2))
        if self.n2 > 0:
            attn = custom_attention(
                qs[1].transpose(1, 2), 
                ks[1].transpose(1, 2), 
                vs[1].transpose(1, 2),
                curvature=self.hyp_curvature, 
                mode='hyp', 
                dropout=self.attn_dropout,
                )
            outputs.append(attn.transpose(1, 2))
        if self.n3 > 0:
            attn = custom_attention(
                qs[2].transpose(1, 2), 
                ks[2].transpose(1, 2), 
                vs[2].transpose(1, 2), 
                mode='sph', 
                dropout=self.attn_dropout, 
                scale=math.sqrt(self.head_dim),
                )
            outputs.append(attn.transpose(1, 2))

        x = torch.cat(outputs, dim=2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(x))


class MultiGeometryBlock(nn.Module):
    def __init__(self, config: MultiGeomGPTConfig):
        super().__init__()
        # self.ln1 = RMSNorm(config.n_embd)
        self.attn = MultiGeometryAttention(config)
        # self.ln2 = RMSNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(p=config.dropout)
        )

    def forward(self, x: torch.Tensor):
        # x = x + self.attn(self.ln1(x))
        # x = x + self.mlp(self.ln2(x))
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


class JointGeometryMLP(nn.Module):
    def __init__(self, config: MultiGeomGPTConfig, geom_type: str):
        """
        :param geom_type: geometry type - can be euc, hyp or sph
        """
        super().__init__()
        self.geom_type = geom_type
        self.n_embd = config.n_embd

        self.uv_proj = nn.Linear(self.n_embd, 2 * 4 * self.n_embd, bias=False)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(4 * self.n_embd, self.n_embd, bias=False)

        if geom_type == "sph":
            self._init_spherical_params()

    def _init_spherical_params(self):
        self.init_value = 1.0
        self.init_scaling = 1.0
        self.suv = nn.Parameter(self.init_scaling * torch.ones(2 * 4 * self.n_embd,))
    
    def forward(self, x: torch.Tensor):
        uv = self.uv_proj(x)
        if self.geom_type == "sph":
            scaling = (self.init_value / self.init_scaling) * (self.n_embd ** 0.5)
            uv = self.suv * scaling * uv

        u, v = torch.chunk(uv, 2, dim=-1)
        h = u * self.act(v)
        h = self.out_proj(h)

        return h


class JointGeometryAttention(nn.Module):
    def __init__(self, config: MultiGeomGPTConfig, geom_type: str):
        """
        :param geom_type: geometry type - can be euc, hyp or sph
        """
        super().__init__()
        self.config = config
        self.geom_type = geom_type
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd

        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

        if geom_type == "hyp":
            self._init_hyperbolic_params()
        elif geom_type == "sph":
            self._init_spherical_params()

    def _init_hyperbolic_params(self):
        if self.config.attn_k_lr:
            x = torch.randn(1, self.n_heads, 1, 1, device=self.qkv.weight.device)
            init_k = torch.exp(x) * self.config.curvature
            self.hyp_curvature = nn.Parameter(init_k)
        else:
            self.register_buffer('hyp_curvature', torch.full((1, self.n_heads, 1, 1), self.config.curvature))
    
    def _init_spherical_params(self):
        self.sqk_init_value = 1.0
        self.sqk_init_scaling = 1.0 / (self.n_embd ** 0.5)
        self.sqk = nn.Parameter(self.sqk_init_scaling * torch.ones(self.n_embd, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim) for t in qkv]

        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if self.geom_type == "euc":
            attn = custom_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mode="euc", scale=1 / (self.n_embd ** 0.5))
        elif self.geom_type == "hyp":
            attn = custom_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mode="hyp", curvature=self.hyp_curvature)
        elif self.geom_type == "sph":
            sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(1, 1, self.n_heads, self.head_dim)
            q = sqk * F.normalize(q, p=2.0, dim=-1)
            k = sqk * F.normalize(k, p=2.0, dim=-1)
            attn = custom_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mode="euc", scale=self.n_embd ** 0.5)

        attn = attn.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn)


class JointGeometryBlock(nn.Module):
    def __init__(self, config: MultiGeomGPTConfig, geom_type: str):
        """
        :param geom_type: geometry type - can be euc, hyp or sph
        """
        super().__init__()
        assert geom_type in ["sph", "hyp", "euc"], f"Unknown geometry was given: {geom_type}"

        self.geom_type = geom_type
        self.n_embd = config.n_embd

        self.attn = JointGeometryAttention(config, geom_type)
        self.mlp = JointGeometryMLP(config, geom_type)

        if geom_type == "sph":
            self._init_spherical_params()

    def _init_spherical_params(self):
        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = 1.0 / (self.n_embd ** 0.5)
        self.attn_alpha = nn.Parameter(self.attn_alpha_init_scaling * torch.ones(self.n_embd))

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = 1.0 / (self.n_embd ** 0.5)
        self.mlp_alpha = nn.Parameter(self.mlp_alpha_init_scaling * torch.ones(self.n_embd))

    def forward(self, x: torch.Tensor):
        if self.geom_type in ["euc", "hyp"]:
            h = F.rms_norm(x, (x.size(-1),))
        else:
            h = x

        h_attn = self.attn(h)

        if self.geom_type == "sph":
            x = self._sph_skip_connection(x, h_attn, "attn")
        else:
            x = x + h_attn

        if self.geom_type in ["euc", "hyp"]:
            h = F.rms_norm(x, (x.size(-1),))
        else:
            h = x

        h_mlp = self.mlp(h)

        if self.geom_type == "sph":
            x = self._sph_skip_connection(x, h_mlp, "mlp")
        else:
            x = x + h_mlp

        return x

    def _sph_skip_connection(self, x: torch.Tensor, h: torch.Tensor, place: str):
        if place == "attn":
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
        else:
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = F.normalize(x, p=2.0, dim=-1)
        B_norm = F.normalize(h, p=2.0, dim=-1)

        return F.normalize(A_norm + lr * (B_norm - A_norm), p=2.0, dim=-1)


class LorentzMLR(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model
    """

    def __init__(self,config: MultiGeomGPTConfig):
        super().__init__()
        if config.head_k_lr > 0:
            self.k = nn.Parameter(torch.tensor(config.curvature))
        else:
            self.register_buffer('k', torch.tensor(config.curvature))
        self.a = torch.nn.Parameter(torch.zeros(config.vocab_size, ))  # optimize properly
        self.z = torch.nn.Parameter(F.pad(torch.zeros(config.vocab_size, config.n_embd - 2), pad=(1, 0), value=1))

        self.init_weights()

    def forward(self, x: torch.Tensor):
        # x: (B, T, num_features)

        # Hyperplane parameters
        sqrt_mK = 1 / self.k.sqrt()  # scalar
        norm_z = torch.norm(self.z, dim=-1)  # (num_classes,)
        w_t = torch.sinh(sqrt_mK * self.a) * norm_z  # (num_classes,)
        w_s = torch.cosh(sqrt_mK * self.a).unsqueeze(-1) * self.z  # (num_classes, num_features - 1)

        beta = torch.sqrt(-w_t ** 2 + torch.norm(w_s, dim=-1) ** 2)  # (num_classes,)

        x0 = x.narrow(-1, 0, 1)  # (B, T, 1)
        x_rest = x.narrow(-1, 1, x.shape[-1] - 1)  # (B, T, num_features -1)
        inner_prod = torch.matmul(x_rest, self.z.T)  # (B, T, num_classes)
        alpha = -x0 * w_t.view(1, 1, -1) + torch.cosh(sqrt_mK * self.a).view(1, 1,
                                                                             -1) * inner_prod  # (B, T, num_classes)
        sqrt_mK_alpha_over_beta = sqrt_mK * alpha / beta.view(1, 1, -1)
        d = self.k.sqrt() * torch.abs(torch.asinh(sqrt_mK_alpha_over_beta))  # (B, T, num_classes)

        logits = torch.sign(alpha) * beta.view(1, 1, -1) * d  # (B, T, num_classes)

        return logits

    def init_weights(self):
        stdv = 1. / math.sqrt(1 + self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)


class MultiGeometryGPT(nn.Module):
    def __init__(self, config: MultiGeomGPTConfig):
        super().__init__()
        self.config = config
        wte = nn.Embedding(config.vocab_size, config.n_embd)

        if config.multi_geom_block:
            layers = nn.ModuleList([MultiGeometryBlock(config) for _ in range(config.n_layers)])
        else:
            geom_dict = {"euc": config.n_euc_layers, "hyp": config.n_hyp_layers, "sph": config.n_sph_layers}
            layers_list = []
            for geom_type in config.layers_order:
                n_layers = geom_dict[geom_type]
                layers_list.extend([JointGeometryBlock(config, geom_type) for _ in range(n_layers)])
            layers = nn.ModuleList(layers_list)

        self.transformer = nn.ModuleDict(dict(
            wte=wte,
            layers=layers
        ))
        
        if config.lm_head_mode == 'euc':
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
            self._init_weights(self.lm_head)
        elif config.lm_head_mode == 'hyp':
            self.lm_head = LorentzMLR(config)
        elif config.lm_head_mode == 'sph':
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
            self._init_weights(self.lm_head)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        x = self.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))

        for block in self.transformer.layers:
            x = block(x)

        # x = self.transformer.ln_f(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
    
    def normalize_sph_matrices(self):
        for layer in self.transformer.layers:
            if layer.geom_type == "sph":
                layer.attn.qkv.weight.data.copy_(F.normalize(layer.attn.qkv.weight.data, p=2.0, dim=-1))
                layer.attn.out_proj.weight.data.copy_(F.normalize(layer.attn.out_proj.weight.data, p=2.0, dim=-1))

                layer.mlp.uv_proj.weight.data.copy_(F.normalize(layer.mlp.uv_proj.weight.data, p=2.0, dim=-1))
                layer.mlp.out_proj.weight.data.copy_(F.normalize(layer.mlp.out_proj.weight.data, p=2.0, dim=-1))

    def generate(self, context: torch.Tensor, max_length: int, temperature: float = 1.0, top_k: int = None):
        self.eval()
        generated = context.clone()
        for _ in range(max_length):
            with torch.no_grad():
                logits, _ = self(generated[:, -self.config.context_length:])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated
    
    def model_size(self):
        """Calculate the model size in millions or thousands, based on parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        if total_params >= 1e6:
            return f"{total_params / 1e6:.2f}M"
        else:
            return f"{total_params / 1e3:.2f}K"


if __name__ == "__main__":
    config = MultiGeomGPTConfig(multi_geom_block=False)
    model = MultiGeometryGPT(config=config)
    x = torch.randint(low=0, high=config.vocab_size - 1, size=(16, config.context_length))
    result = model(x)