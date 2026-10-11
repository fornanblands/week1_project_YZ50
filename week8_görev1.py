import torch
import torch.nn as nn
from torch.nn import functional as F

# ###########################################################################
#                          HAFTA 8 — GÖREV 1
#   Tek head'i multi-head attention'a çevir: aynı boyutu (n_embd = 32)
#   n_head tane küçük head'e böl, çıktıları birleştir (concat).
# ###########################################################################

# ---------------------------------------------------------------------------
# hiperparametreler (Hafta 7 ile aynı)
batch_size    = 32
block_size    = 8
max_iters     = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters    = 200
n_embd        = 32
n_head        = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH     = '/Users/mertisler/Desktop/YZ50/Tiny_Shakespeare.py'
BASELINE_PATH = '/Users/mertisler/Desktop/YZ50/baseline.txt'
# ---------------------------------------------------------------------------

assert n_embd % n_head == 0, "n_embd, n_head'e tam bölünmeli"

torch.manual_seed(1337)

# ===========================================================================
# 1) Veri (Hafta 7 ile aynı)
# ===========================================================================
text = open(DATA_PATH, 'r', encoding='utf-8').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ===========================================================================
# 2) Head (Hafta 7'deki sınıfın aynısı)
# ===========================================================================
class Head(nn.Module):
    """Tek bir self-attention head."""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.last_wei = None

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, hs)
        q = self.query(x)   # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5           # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.last_wei = wei.detach()
        v = self.value(x)   # (B, T, hs)
        return wei @ v      # (B, T, hs)

# ===========================================================================
# 3) YENİ: MultiHeadAttention 
#   Her head kendi key/query/value'su ile BAĞIMSIZ çalışır, kendi wei'sini
#   hesaplar. Çıktılar kanal boyutunda (dim=-1) yan yana yapıştırılır:
#     4 head x (B, T, 8)  ->  cat  ->  (B, T, 32)
#   Çıktı boyutu tek head'deki ile aynı (n_embd), lm_head değişmiyor.

# ===========================================================================
class MultiHeadAttention(nn.Module):
    """Paralel çalisan birden çok self-attention head."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, num_heads*head_size)
# ===========================================================================
# 4) attention katmanı dışarıdan veriliyor.
#    Böylece tek head ve multi-head modeli AYNI iskeleti kullanıyor,
#    aradaki tek fark attention katmanı oluyor.
# ===========================================================================
class AttentionLanguageModel(nn.Module):

    def __init__(self, sa):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa      = sa                              # Head(32) veya MultiHeadAttention(4, 8)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                    # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb
        x = self.sa(x)                # (B, T, n_embd)
        logits = self.lm_head(x)      # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def train(model, name):
    print(f"\n--- {name} eğitiliyor ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return estimate_loss(model)

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# ===========================================================================
# 5) Tek head (Hafta 7 tekrarı)
# ===========================================================================
torch.manual_seed(1337)
single = AttentionLanguageModel(Head(n_embd)).to(device)
print("tek head parametre sayısı  :", n_params(single))

# ===========================================================================
# 6) Multi-head: aynı 32 boyut, 4 x 8'e bölünmüş
#    Parametre sayısı AYNI:
#      tek head : 3 * (32x32)       = 3072
#      4 head   : 4 * 3 * (32x8)    = 3072
#    Kapasite artmadı; aynı parametreler 4 bağımsız kanala bölündü.
# ===========================================================================
torch.manual_seed(1337)
multi = AttentionLanguageModel(MultiHeadAttention(n_head, n_embd // n_head)).to(device)
print("multi-head parametre sayısı:", n_params(multi))

# shape kontrolü
_xb, _ = get_batch('train')
_x = multi.token_embedding_table(_xb) + multi.position_embedding_table(
    torch.arange(block_size, device=device))
print("tek head çıktısı :", tuple(multi.sa.heads[0](_x).shape), " (B, T, 8)")
print("birleşik çıktı   :", tuple(multi.sa(_x).shape), " (B, T, 32)")

final_single = train(single, f"1 head, hs={n_embd}")
final_multi  = train(multi,  f"{n_head} head, hs={n_embd // n_head}")

# ===========================================================================
# 7) Yan yana
# ===========================================================================
try:
    with open(BASELINE_PATH) as f:
        bigram_val = float(f.readline().split()[-1])   # "bigram val_loss 2.4xxx"
except (FileNotFoundError, ValueError):
    bigram_val = None

single_val = final_single['val'].item()
multi_val  = final_multi['val'].item()

print("\n" + "-" * 50)
print(f"{'model':<30}{'val loss':>10}")
print("-" * 50)
if bigram_val is not None:
    print(f"{'Bigram (H7 Görev 1)':<30}{bigram_val:>10.4f}")
print(f"{f'1 head, hs={n_embd}':<30}{single_val:>10.4f}")
print(f"{f'{n_head} head, hs={n_embd // n_head}':<30}{multi_val:>10.4f}")
print("-" * 50)
print(f"tek head -> multi-head düşüş: {single_val - multi_val:.4f}  "
      f"({100 * (single_val - multi_val) / single_val:.1f}%)")

# ===========================================================================
# 8) Her head 'First Ci'nin son karakterinde nereye bakıyor?
# ===========================================================================
idx = train_data[:block_size].unsqueeze(0).to(device)
labels = [repr(itos[i]) for i in idx[0].tolist()]
multi.eval()
with torch.no_grad():
    multi(idx)
multi.train()
print("\nörnek dizi:", repr(decode(idx[0].tolist())))
for h_i, h in enumerate(multi.sa.heads):
    w = h.last_wei[0, -1].cpu()
    dagilim = "  ".join(f"{labels[j]}:{w[j].item():.2f}" for j in range(block_size))
    print(f"head {h_i}: {dagilim}   -> en çok {labels[w.argmax().item()]}")

# ===========================================================================
# 9) Örnek metin
# ===========================================================================
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nÜretilen metin (multi-head):\n")
print(decode(multi.generate(context, max_new_tokens=500)[0].tolist()))