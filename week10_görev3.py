# Görev 3 — Modeli konuştur (videoda 0:31:00 - 0:45:50, build-nanogpt commit af6e591 + ce84eb3)
# Çalıştırma:  ~/.venvs/main/bin/python ~/Desktop/YZ50/week10/week10_gorev3.py
# Gerekenler:  uv pip install --python ~/.venvs/main torch transformers tiktoken

import tiktoken

# ---------------------------------------------------------------------------
# GPT-2 modeli (Görev 2 ile aynı yapı, build-nanogpt commit af6e591) — dosya tek başına çalışır
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

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
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

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

    def forward(self, idx):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# ---------------------------------------------------------------------------
# 1) Cihaz seçimi (Mac'te "mps" seçilecek)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

enc = tiktoken.get_encoding("gpt2")


# ---------------------------------------------------------------------------
# 2) Sampling döngüsü (top-k)
@torch.no_grad()
def generate(model, prompt, num_return_sequences=5, max_length=30, top_k=50, seed=42):
    """prompt'u num_return_sequences kez kopyalar, toplam uzunluk max_length token olana kadar
    her adımda son pozisyonun logits'inden top-k sampling ile bir token ekler."""
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long)              # (T,)
    x = x.unsqueeze(0).repeat(num_return_sequences, 1)      # (B, T)
    x = x.to(device)

    torch.manual_seed(seed)                                 # tekrarlanabilir örnekler (videodaki gibi 42)
    while x.size(1) < max_length:
        x_cond = x[:, -model.config.block_size:]            # 1024'ü aşarsa kırp
        logits = model(x_cond)                               # (B, T, vocab_size)
        if isinstance(logits, tuple):                        # forward (logits, loss) döndürüyorsa
            logits = logits[0]
        logits = logits[:, -1, :]                            # sadece son pozisyon: (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        # top-k: en olası k token'ı tut, gerisini sıfırla (HF pipeline varsayılanı k=50)
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)   # (B, k), (B, k)
        # multinomial girdinin toplamının 1 olmasını şart koşmaz, yeniden normalize etmeye gerek yok
        ix = torch.multinomial(topk_probs, 1)            # (B, 1) -> k içindeki sıra
        xcol = torch.gather(topk_indices, -1, ix)                     # (B, 1) -> gerçek token id
        x = torch.cat((x, xcol), dim=1)

    print(f"\n=== prompt: {prompt!r}  ({len(tokens)} token) ===")
    for i in range(num_return_sequences):
        text = enc.decode(x[i, :max_length].tolist())
        print(">", text)


# ---------------------------------------------------------------------------
# 3) Token analizi: aynı anlamdaki cümle kaç token tutuyor?
def token_report(s):
    ids = enc.encode(s)
    words = len(s.split())
    pieces = [enc.decode_single_token_bytes(i) for i in ids]
    print(f"\n{s!r}")
    print(f"  karakter={len(s)}  utf8_byte={len(s.encode())}  kelime={words}  "
          f"token={len(ids)}  token/kelime={len(ids)/words:.2f}  karakter/token={len(s)/len(ids):.2f}")
    print("  parçalar:", pieces)


if __name__ == "__main__":
    model = GPT.from_pretrained("gpt2")
    model.eval()
    model.to(device)

    # İngilizce (videodaki prompt) ve Türkçe karşılığı
    generate(model, "Hello, I'm a language model,")
    generate(model, "Merhaba, ben bir dil modeliyim,")
    generate(model, "Türkiye'nin başkenti Ankara'dır ve en kalabalık şehri İstanbul'dur.", max_length=60)

    print("\n" + "=" * 70 + "\nTOKEN SAYILARI")
    for s in ["Hello, I'm a language model,",
              "Merhaba, ben bir dil modeliyim,",
              "The capital of Turkey is Ankara and its most populous city is Istanbul.",
              "Türkiye'nin başkenti Ankara'dır ve en kalabalık şehri İstanbul'dur."]:
        token_report(s)

    # Kontrol: rastgele başlatılmış model (commit ce84eb3) -> anlamsız çıktı beklenir
    print("\n" + "=" * 70 + "\nRASTGELE AĞIRLIKLI MODEL")
    rand_model = GPT(GPTConfig()).eval().to(device)
    generate(rand_model, "Hello, I'm a language model,")