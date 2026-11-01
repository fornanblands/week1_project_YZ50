import argparse
import inspect
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # Weight decay yalnız 2 boyutlu parametrelere (matmul ağırlıkları + embedding'ler) uygulanıyor,
        # bias'lar ve LayerNorm parametreleri decay almıyor.
        # Sonuç: 50 tensör / 124,354,560 parametreye uygulandı, 98 tensör / 121,344 parametreye uygulanmadı.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        print(f"using fused AdamW: {use_fused}")
        # AdamW: GPT-3 makalesi Ek B'deki beta'lar ve epsilon
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8,
                                 **({'fused': True} if use_fused else {}))

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T, path='input.txt'):
        self.B, self.T = B, T
        with open(path, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text))
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} micro-batches")
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

# warmup'lı cosine learning rate schedule
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def plot_lr(path="lr_schedule.png", extra=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = list(range(max_steps + extra))
    lrs = [get_lr(s) for s in steps]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(steps, lrs, color="#3b6fb6", lw=2)
    ax.axvline(warmup_steps, color="#999", ls="--", lw=1)
    ax.axvline(max_steps, color="#999", ls="--", lw=1)
    ax.text(warmup_steps + 0.5, max_lr * 1.01, "warmup biter", fontsize=9, color="#555")
    ax.text(max_steps + 0.5, min_lr * 1.6, "min_lr = %10", fontsize=9, color="#555")
    ax.set_xlabel("adım")
    ax.set_ylabel("learning rate")
    ax.set_title(f"Warmup + cosine decay (max_lr={max_lr}, warmup={warmup_steps}, max_steps={max_steps})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"{path} yazıldı")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=None, help="micro batch size")
    p.add_argument("--T", type=int, default=None, help="sequence length")
    p.add_argument("--total-batch", type=int, default=None, help="adım başına toplam token")
    p.add_argument("--plot-lr", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16"], default="auto",
                   help="auto: Ampere+ GPU'da bf16, T4 gibi eski GPU'da fp16")
    p.add_argument("--max-steps", type=int, default=50, help="T4'te birkaç yüz adım, örn. 300")
    p.add_argument("--warmup-steps", type=int, default=10)
    args = p.parse_args()
    max_steps, warmup_steps = args.max_steps, args.warmup_steps

    globals().update(max_steps=max_steps, warmup_steps=warmup_steps)
    if args.plot_lr:
        plot_lr()
        raise SystemExit

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    torch.manual_seed(1337)
    if device == "cuda":
        torch.cuda.manual_seed(1337)

    B = args.B or (16 if device == "cuda" else 4)
    T = args.T or (1024 if device == "cuda" else 512)
    total_batch_size = args.total_batch or (524288 if device == "cuda" else 16384)
    assert total_batch_size % (B * T) == 0, "total_batch_size, B*T'ye tam bölünmeli"
    # gradient accumulation: büyük batch = grad_accum_steps tane küçük batch
    grad_accum_steps = total_batch_size // (B * T)
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T)
    torch.set_float32_matmul_precision('high')

    model = GPT(GPTConfig())
    model.to(device)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)

    use_amp = not args.no_amp and device in ("cuda", "mps")
    if args.dtype == "auto":
        bf16_native = device != "cuda" or torch.cuda.get_device_capability()[0] >= 8
        amp_dtype = torch.bfloat16 if bf16_native else torch.float16
    else:
        amp_dtype = getattr(torch, args.dtype)
    scaler = torch.amp.GradScaler(device) if (use_amp and amp_dtype == torch.float16) else None
    print(f"autocast: {amp_dtype if use_amp else 'off'} | GradScaler: {scaler is not None}")
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if use_amp:
                with torch.autocast(device_type=device, dtype=amp_dtype):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
            # Neden loss'u accumulation adım sayısına bölüyoruz?
            # cross_entropy her küçük batch'in içinde ortalama alıyor, ama backward() her çağrıda
            # gradyanı .grad'ın üstüne ekliyor. 8 küçük batch'i bölmeden geri yayarsak 8 ortalamanın
            # toplamını elde ederiz, yani büyük batch'in gradyanının 8 katını. Bize toplam değil ortalama lazım.
            # Her küçük batch'in loss'unu 8'e bölünce toplam, büyük batch'in tek seferde vereceği
            # gradyanla birebir aynı oluyor.
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if scaler is not None:
            scaler.unscale_(optimizer)
        # gradient clipping: global norm 1.0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        tokens_per_sec = B * T * grad_accum_steps / dt
        print(f"step {step:4d} | loss: {float(loss_accum):.6f} | lr {lr:.4e} | norm: {norm:.4f} "
              f"| dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
