import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# 1) Konfigürasyon — GPT-2 small (124M)
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024   # maksimum bağlam uzunluğu (wpe'nin satır sayısı)
    vocab_size: int = 50257  # 50.000 BPE birleştirmesi + 256 bayt token + 1 <|endoftext|>
    n_layer: int = 12        # blok sayısı
    n_head: int = 12         # attention kafası sayısı
    n_embd: int = 768        # embedding boyutu (kafa başına 768 / 12 = 64)


# ---------------------------------------------------------------------------
# 2) Causal self-attention — q, k, v TEK katmanda, kafalar ayrı bir boyutta
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # q, k, v projeksiyonları tek bir Linear'da yan yana: 768 -> 3*768 = 2304
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # kafaların birleşik çıktısını tekrar residual stream'e yansıtan projeksiyon
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Nedensel maske. Öğrenilmeyen bir buffer. Adı OpenAI/HF ile aynı olsun diye
        # "bias" (kafa karıştırıcı ama HF böyle adlandırıyor). Parametre değildir.
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, zaman (token sayısı), kanal (n_embd)

        qkv = self.c_attn(x)                         # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)      # her biri (B, T, C)

        # Kafaları ayrı bir "batch" boyutu yap: (B, T, C) -> (B, nh, T, hs)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # softmax(Q K^T / sqrt(hs) + maske) V  — tüm kafalar paralel
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))        # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                                                     # (B, nh, T, hs)

        # Kafaları yan yana birleştir: (B, nh, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


# ---------------------------------------------------------------------------
# 3) MLP — 4x genişle, GELU (tanh yaklaşımı), daral
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GPT-2 tanh yaklaşımlı GELU ile eğitildi (HF'de adı "gelu_new").
        # Ağırlıkları birebir kullanmak için aynı fonksiyonu kullanmalıyız.
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# ---------------------------------------------------------------------------
# 4) Blok — pre-LayerNorm + residual bağlantılar
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # iletişim (token'lar arası)
        x = x + self.mlp(self.ln_2(x))   # hesaplama (her token kendi içinde)
        return x


# ---------------------------------------------------------------------------
# 5) GPT — isimler HF'in şemasıyla birebir aynı (transformer.wte, transformer.h.0...)
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # token embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),   # pozisyon embedding
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),                     # son LayerNorm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        # idx: (B, T) token id'leri
        B, T = idx.size()
        assert T <= self.config.block_size, f"T={T} > block_size={self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.transformer.wpe(pos)   # (T, C)
        tok_emb = self.transformer.wte(idx)   # (B, T, C)
        x = tok_emb + pos_emb                 # yayın (broadcast) ile (B, T, C)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)              # (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type, transpose=True):
        """HF GPT-2 ağırlıklarını bizim modelimize kopyalar.
        transpose=False yalnızca deney içindir (neden transpoze gerektiğini görmek için)."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print(f"önceden eğitilmiş GPT-2 yükleniyor: {model_type}")

        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }[model_type]
        config_args["vocab_size"] = 50257  # tüm GPT-2'lerde aynı
        config_args["block_size"] = 1024   # tüm GPT-2'lerde aynı
        model = GPT(GPTConfig(**config_args))

        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]  # maske buffer'ı: kopyalanmaz

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith(".attn.masked_bias")   # buffer
                      and not k.endswith(".attn.bias")]        # buffer (maske)

        # HF bu dört ağırlığı Conv1D ile (giriş, çıkış) düzeninde saklar;
        # nn.Linear ise (çıkış, giriş) bekler -> transpoze ederek kopyalarız.
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight",
                      "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys), \
            f"anahtar sayısı uyuşmuyor: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, k
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t() if transpose else sd_hf[k])
            else:
                assert sd_hf[k].shape == sd[k].shape, k
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


# ---------------------------------------------------------------------------
# 6) Doğrulama: aynı girdi -> aynı logits
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tiktoken
    from transformers import GPT2LMHeadModel

    torch.manual_seed(0)

    model = GPT.from_pretrained("gpt2").eval()
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()  # eval: HF'deki dropout (0.1) kapalı

    n_params = sum(p.numel() for p in model.parameters())
    print(f"bizim modelin parametre sayısı: {n_params:,}")
    # Not: weight tying yapmadığımız için lm_head ayrıca sayılır -> 124.439.808 + 38.597.376

    enc = tiktoken.get_encoding("gpt2")
    text = "Hello, I'm a language model,"
    idx = torch.tensor(enc.encode(text), dtype=torch.long).unsqueeze(0)  # (1, T)
    print(f"girdi: {text!r} -> token'lar: {idx.tolist()[0]}")

    with torch.no_grad():
        logits_ours = model(idx)              # (1, T, 50257)
        logits_hf = model_hf(idx).logits      # (1, T, 50257)

    diff = (logits_ours - logits_hf).abs()
    print(f"logits boyutu (bizim / HF): {tuple(logits_ours.shape)} / {tuple(logits_hf.shape)}")
    print(f"maksimum mutlak fark      : {diff.max().item():.3e}")
    print(f"ortalama mutlak fark      : {diff.mean().item():.3e}")
    print(f"allclose (atol=1e-4)      : {torch.allclose(logits_ours, logits_hf, atol=1e-4)}")
    print(f"her pozisyonda argmax aynı: "
          f"{torch.equal(logits_ours.argmax(-1), logits_hf.argmax(-1))}")
    print("son pozisyon, ilk 5 logit (bizim):", logits_ours[0, -1, :5])
    print("son pozisyon, ilk 5 logit (HF)   :", logits_hf[0, -1, :5])

    next_ids = logits_ours[0, -1].topk(5).indices.tolist()
    print("en olası 5 sonraki token:", [enc.decode([i]) for i in next_ids])

    # --- Deney: transpoze ETMEZSEK ne olur? ---------------------------------
    # c_attn ve c_fc kare olmadığı için assert hemen patlar. Ama attn.c_proj
    # (768x768) KARE: boyut kontrolü bu hatayı yakalayamaz. Sessizce yanlış yüklenir.
    # Bunu göstermek için yalnızca kare matrisleri transpoze etmeden yükleyelim:
    print("\n[deney] kare attn.c_proj'u transpoze etmeden yüklersek:")
    bozuk = GPT.from_pretrained("gpt2").eval()
    sd_hf = model_hf.state_dict()
    with torch.no_grad():
        for i in range(bozuk.config.n_layer):
            k = f"transformer.h.{i}.attn.c_proj.weight"
            bozuk.state_dict()[k].copy_(sd_hf[k])   # .t() YOK
        logits_bozuk = bozuk(idx)
    print(f"  maksimum mutlak fark: {(logits_bozuk - logits_hf).abs().max().item():.3e}")
    print("  en olası 5 sonraki token:",
          [enc.decode([i]) for i in logits_bozuk[0, -1].topk(5).indices.tolist()])