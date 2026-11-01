import argparse
import gc
import math
import platform
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
        self.flash = config.flash
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                         .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            # flash attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
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
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    flash: bool = False

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
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T, path='input.txt'):
        self.B, self.T = B, T
        with open(path, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text))
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


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

def empty_cache(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

def hardware_name(device):
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return f"Apple {platform.machine()} (MPS), macOS {platform.mac_ver()[0]}"
    return f"CPU {platform.processor() or platform.machine()}"

def tf32_support(device):
    if device != "cuda":
        return False, "TF32 yalnızca NVIDIA Ampere+ GPU'larda var; bu cihazda karşılığı yok"
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False, f"GPU compute capability {major}.x < 8.0, TF32 tensor core yok"
    return True, ""

def autocast_support(device, dtype):
    if device == "cuda" and dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        major, minor = torch.cuda.get_device_capability()
        return False, f"bfloat16 donanımda yok (compute capability {major}.{minor} < 8.0); --dtype float16 kullan"
    try:
        a = torch.randn(64, 64, device=device)
        with torch.autocast(device_type=device, dtype=dtype):
            b = a @ a
        synchronize(device)
        if b.dtype != dtype:
            return False, f"autocast {dtype} çalıştı ama matmul {b.dtype} kaldı"
        return True, ""
    except Exception as e:
        return False, f"autocast {dtype} desteklenmiyor: {type(e).__name__}: {str(e).splitlines()[0][:120]}"


def run_stage(cfg, args, device, loader):
    torch.manual_seed(1337)
    if device == "cuda":
        torch.cuda.manual_seed(1337)
    # TF32
    torch.set_float32_matmul_precision('high' if cfg["tf32"] else 'highest')

    model = GPT(GPTConfig(vocab_size=cfg["vocab"], flash=cfg["flash"]))
    model.to(device)
    if cfg["compile"]:
        # torch.compile
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    use_amp = cfg["amp"] is not None
    scaler = None
    if use_amp and cfg["amp"] == torch.float16:
        scaler = torch.amp.GradScaler(device)

    loader.current_position = 0
    dts, losses = [], []
    for i in range(args.steps):
        t0 = time.time()
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if use_amp:
            # bfloat16 / float16
            with torch.autocast(device_type=device, dtype=cfg["amp"]):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        synchronize(device)
        t1 = time.time()
        dt = t1 - t0
        tps = loader.B * loader.T / dt
        losses.append(loss.item())
        print(f"  step {i:3d} | loss: {loss.item():.6f} | dt: {dt*1000:8.2f}ms | tok/sec: {tps:10.2f}")
        if i >= args.warmup:
            dts.append(dt)

    del model, optimizer
    if cfg["compile"]:
        torch._dynamo.reset()
    empty_cache(device)
    mean_dt = sum(dts) / len(dts)
    return mean_dt, loader.B * loader.T / mean_dt, losses[-1]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--warmup", type=int, default=5, help="ortalamaya katılmayan ilk adımlar (compile süresi dahil)")
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--only", type=str, default=None,
                   help="virgüllü aşama listesi, örn. baseline,flash,vocab (varsayılan: hepsi)")
    args = p.parse_args()
    assert args.steps > args.warmup

    device = get_device()
    if args.B is None:
        args.B = 16 if device == "cuda" else 4
    if args.T is None:
        args.T = 1024 if device == "cuda" else 512
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    hw = hardware_name(device)
    print(f"device: {device} | {hw} | torch {torch.__version__}")
    print(f"B={args.B}, T={args.T}, tokens/step={args.B*args.T}, steps={args.steps}, warmup={args.warmup}")
    loader = DataLoaderLite(B=args.B, T=args.T)

    stages = [
        ("baseline", "fp32 başlangıç"),
        ("tf32",     "+ TF32 matmul"),
        ("amp",      f"+ {args.dtype} autocast"),
        ("compile",  "+ torch.compile"),
        ("flash",    "+ flash attention"),
        ("vocab",    "+ vocab 50304"),
    ]
    only = set(args.only.split(",")) if args.only else None

    cfg = dict(tf32=False, amp=None, compile=False, flash=False, vocab=50257)
    rows = []
    base_dt = None
    prev_dt = None

    for key, label in stages:
        skip_reason = ""
        trial = dict(cfg)
        if key == "tf32":
            ok, skip_reason = tf32_support(device)
            if ok: trial["tf32"] = True
        elif key == "amp":
            ok, skip_reason = autocast_support(device, amp_dtype)
            if ok: trial["amp"] = amp_dtype
        elif key == "compile":
            trial["compile"] = True
        elif key == "flash":
            trial["flash"] = True
        elif key == "vocab":
            # vocab 50257 -> 50304
            trial["vocab"] = 50304

        if only is not None and key not in only and key != "baseline":
            continue

        if skip_reason:
            print(f"\n[{label}] ATLANDI: {skip_reason}")
            rows.append((label, None, None, None, None, "atlandı: " + skip_reason))
            continue

        print(f"\n[{label}] {trial}")
        try:
            dt, tps, last_loss = run_stage(trial, args, device, loader)
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e).strip().splitlines()[0][:150] if str(e).strip() else ''}"
            print(f"[{label}] HATA, atlandı: {msg}")
            rows.append((label, None, None, None, None, "atlandı (hata): " + msg))
            if key == "compile":
                torch._dynamo.reset()
            empty_cache(device)
            continue

        cfg = trial
        if base_dt is None:
            base_dt = dt
        step_gain = (prev_dt / dt) if prev_dt else 1.0
        rows.append((label, dt * 1000, tps, step_gain, base_dt / dt, f"son loss {last_loss:.3f}"))
        prev_dt = dt

    lines = [
        f"Donanım: {hw} | torch {torch.__version__} | B={args.B}, T={args.T} "
        f"({args.B*args.T} token/adım) | {args.steps} adım, ilk {args.warmup} hariç ortalama",
        "",
        "| Aşama | dt (ms) | tok/sec | önceki aşamaya göre | başlangıca göre | not |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, dt, tps, sg, tg, note in rows:
        if dt is None:
            lines.append(f"| {label} | – | – | – | – | {note} |")
        else:
            lines.append(f"| {label} | {dt:.1f} | {tps:,.0f} | {sg:.2f}x | {tg:.2f}x | {note} |")
    measured = [r for r in rows if r[1] is not None and r[3] is not None]
    if len(measured) > 1:
        best = max(measured[1:], key=lambda r: r[3])
        lines += ["", f"En çok kazandıran adım: **{best[0]}** ({best[3]:.2f}x)"]
    table = "\n".join(lines)
    print("\n" + table)
    with open("results.md", "w") as f:
        f.write(table + "\n")
    print("\nresults.md yazıldı")

# Sonuçlar (Tesla T4, B=4, T=1024, 15 adım, ilk 5 adım hariç ortalama):
#
# | Aşama              | dt (ms) | tok/sec | önceki aşamaya göre | başlangıca göre |
# |--------------------|---------|---------|---------------------|-----------------|
# | fp32 başlangıç     |  1202.0 |   3,408 | 1.00x               | 1.00x           |
# | + TF32             |       - |       - | atlandı             |                 |
# | + float16          |   474.9 |   8,624 | 2.53x               | 2.53x           |
# | + torch.compile    |   330.0 |  12,412 | 1.44x               | 3.64x           |
# | + flash attention  |   288.3 |  14,207 | 1.14x               | 4.17x           |
# | + vocab 50304      |   283.3 |  14,456 | 1.02x               | 4.24x           |
#
# TF32'yi atladım: T4'ün compute capability'si 7.5, TF32 ise Ampere (8.0) ve sonrasında var.
# bfloat16 da T4'te yok, onun yerine float16 kullandım ve GradScaler ekledim.
#
# Benim donanımımda en çok kazandıran adım float16 oldu (2.53x).
# T4'ün tensor core'ları float16 için tasarlanmış, fp32'de hiç kullanılmıyorlar.
# torch.compile'ın ilk adımı 43 saniye sürdü (derleme), sonrasında 1.44x hızlandırdı.
# Vocab 50304'ün 1.02x'i ölçüm gürültüsü sınırında, T4'te belirgin bir etkisi yok.
# Loss değerleri her aşamada fp32 ile neredeyse aynı kaldı, yani hızlandırma öğrenmeyi bozmadı.
