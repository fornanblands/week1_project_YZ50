import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# ###########################################################################
#                          HAFTA 8 — GÖREV 2
#   Transformer bloğunu parça parça tamamla: feedforward, residual bağlantı,
#   LayerNorm. Her parçayı ekledikten sonra val loss'u kaydet.
#   Tek tablo: bigram, tek head, multi-head, +feedforward, +residual, +LayerNorm.
#   LayerNorm'u Hafta 6'daki BatchNorm1d ile karşılaştır.
#   (1:24:25 - 1:37:49)
#
#   Videodaki sıra:
#     a) multi-head'in arkasına FeedForward      "iletişim -> düşünme"
#     b) Block = sa + ffwd, 3 blok üst üste     derinleşince eğitim zorlaşıyor
#     c) residual bağlantı + proj + 4x ffwd      derin ağ yeniden eğitilebilir
#     d) LayerNorm (pre-norm) + son ln_f         
#
#   Görev 1 dosyasından TEK fark eden sınıflar: MultiHeadAttention'a opsiyonel
#   proj eklendi (varsayılan kapalı -> Görev 1 ile birebir aynı davranır),
#   FeedForward ve Block yeni. AttentionLanguageModel hiç değişmedi: ortadaki
#   "sa" katmanına artık blok dizisini veriyoruz.
# ###########################################################################

# ---------------------------------------------------------------------------
# hiperparametreler (Görev 1 ile aynı; n_layer yeni)
batch_size    = 32
block_size    = 8
max_iters     = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters    = 200
n_embd        = 32
n_head        = 4
n_layer       = 3        # videodaki gibi 3 blok
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH     = '/Users/mertisler/Desktop/YZ50/Tiny_Shakespeare.py'
BASELINE_PATH = '/Users/mertisler/Desktop/YZ50/baseline.txt'
RESULTS_PATH  = '/Users/mertisler/Desktop/YZ50/week8/week8_gorev2_sonuclar.txt'
RUN_DEEP_NO_RESIDUAL = True   # (b) ara adımı: 3 blok ama residual yok. False -> atla
# ---------------------------------------------------------------------------

assert n_embd % n_head == 0, "n_embd, n_head'e tam bölünmeli"

torch.manual_seed(1337)

# ===========================================================================
# 1) Veri (Görev 1 ile aynı)
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
# 2) Head (değişmedi)
# ===========================================================================
class Head(nn.Module):
    """Tek bir self-attention head."""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

# ===========================================================================
# 3) MultiHeadAttention — Görev 1'deki sınıf + opsiyonel proj
#
#   proj (residual aşamasında açılıyor: head'lerin concat
#   çıktısını bir Linear'dan geçirip residual yola "geri yansıtıyor".
#   proj=False iken Görev 1'deki sınıfın birebir aynısı.
# ===========================================================================
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, proj=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) if proj else nn.Identity()

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_embd)
        return self.proj(out)

# ===========================================================================
# 4) FeedForward 
#
#   Attention'da token'lar birbiriyle KONUŞTU (iletişim). Ama topladıkları
#   bilgiyi işlemeye ("düşünmeye") vakit olmadan doğrudan lm_head'e gittiler.
#   FeedForward bunu sağlar: her token'a AYRI AYRI uygulanan küçük bir MLP.
#   Token'lar arasında bilgi taşımaz; nn.Linear (B, T, C) girdinin sadece
#   son boyutuna etki eder -> B*T tane token, aynı ağırlıklarla bağımsız işlenir.
#
#   wide=False : Linear(32, 32) + ReLU                          (videonun ilk hali)
#   wide=True  : Linear(32, 128) + ReLU + Linear(128, 32)        (residual aşaması)
#     - 4x genişlik makaleden: d_model=512, d_ff=2048 (3.3. bölüm)
#     - sondaki Linear, MultiHeadAttention'daki proj ile aynı rolde: residual
#       yola geri yansıtma
# ===========================================================================
class FeedForward(nn.Module):

    def __init__(self, n_embd, wide=False):
        super().__init__()
        if wide:
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(n_embd, n_embd),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.net(x)

# ===========================================================================
# 5) YENİ: Block = iletişim (sa) + hesaplama (ffwd)  
#
#   residual=False :  x = sa(x);         x = ffwd(x)
#   residual=True  :  x = x + sa(x);     x = x + ffwd(x)            
#   layernorm=True :  x = x + sa(ln1(x)); x = x + ffwd(ln2(x))      
#
#   RESIDUAL neden? Toplama işlemi gradyanı OLDUĞU GİBİ iki kola dağıtır.
#   x'ten çıkıp loss'a giden "otoyol" üzerinde gradyan hiçbir katmandan
#   geçmeden en başa kadar ulaşır. sa ve ffwd o otoyoldan ayrılan yan
#   yollar; başlangıçta katkıları küçüktür, eğitimle yavaş yavaş devreye
#   girerler. Böylece derin ağ, sığ ağ kadar kolay optimize edilir.
#
#   PRE-NORM: Makalede LayerNorm sa/ffwd'den SONRA (post-norm). Karpathy
#   (ve günümüzdeki çoğu model) ÖNCE uyguluyor: girdiyi normalize et,
#   dönüştür, otoyola ekle. Otoyolun kendisi normalize edilmiyor.
# ===========================================================================
class Block(nn.Module):

    def __init__(self, n_embd, n_head, residual=False, layernorm=False):
        super().__init__()
        head_size = n_embd // n_head
        self.residual = residual
        self.sa   = MultiHeadAttention(n_head, head_size, proj=residual)
        self.ffwd = FeedForward(n_embd, wide=residual)
        self.ln1  = nn.LayerNorm(n_embd) if layernorm else nn.Identity()
        self.ln2  = nn.LayerNorm(n_embd) if layernorm else nn.Identity()

    def forward(self, x):
        if self.residual:
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            x = self.sa(self.ln1(x))
            x = self.ffwd(self.ln2(x))
        return x

# ===========================================================================
# 6) Dil modeli (Görev 1'deki AttentionLanguageModel, DEĞİŞMEDİ)
#    'sa' artık tek bir head de, multi-head de, blok dizisi de olabilir.
# ===========================================================================
class AttentionLanguageModel(nn.Module):

    def __init__(self, sa):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa      = sa
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.sa(x)
        logits = self.lm_head(x)

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
# 7) Aşamalar
#    Her aşamadan önce aynı seed -> her satır tek başına tekrarlanabilir.
#    Aşamaların tanımı videodakiyle aynı:
#      (a) multi-head + ffwd                 : nn.Sequential(MHA, FeedForward)
#      (b) 3 blok, residual yok              : ara adım, "derinlik tek başına yetmez"
#      (c) 3 blok + residual (+proj, 4x ffwd)
#      (d) (c) + LayerNorm, sona ln_f        : nn.Sequential(*blocks, nn.LayerNorm)
# ===========================================================================
stages = [
    ("1 head (hs=32)",
     lambda: Head(n_embd)),
    ("multi-head (4x8)",
     lambda: MultiHeadAttention(n_head, n_embd // n_head)),
    ("+ feedforward",
     lambda: nn.Sequential(
         MultiHeadAttention(n_head, n_embd // n_head),
         FeedForward(n_embd))),
]
if RUN_DEEP_NO_RESIDUAL:
    stages.append(
        (f"  ({n_layer} blok, residual yok)",
         lambda: nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])))
stages += [
    (f"+ residual ({n_layer} blok)",
     lambda: nn.Sequential(*[Block(n_embd, n_head, residual=True) for _ in range(n_layer)])),
    ("+ LayerNorm",
     lambda: nn.Sequential(
         *[Block(n_embd, n_head, residual=True, layernorm=True) for _ in range(n_layer)],
         nn.LayerNorm(n_embd))),   # ln_f: lm_head'den hemen önce son normalize
]

results = []      # (isim, parametre sayısı, train loss, val loss)
model = None
for name, make_sa in stages:
    torch.manual_seed(1337)
    model = AttentionLanguageModel(make_sa()).to(device)
    print(f"\n{name}: {n_params(model)} parametre")
    t0 = time.time()
    final = train(model, name)
    print(f"süre: {time.time() - t0:.0f} sn")
    results.append((name, n_params(model), final['train'].item(), final['val'].item()))

# ===========================================================================
# 8) Tablo
# ===========================================================================
try:
    with open(BASELINE_PATH) as f:
        bigram_val = float(f.readline().split()[-1])   # "bigram val_loss 2.4xxx"
except (FileNotFoundError, ValueError):
    bigram_val = None

lines = []
lines.append("-" * 66)
lines.append(f"{'model':<30}{'param':>10}{'train':>10}{'val':>10}{'Δval':>6}")
lines.append("-" * 66)
prev = bigram_val
if bigram_val is not None:
    lines.append(f"{'bigram':<30}{'':>10}{'':>10}{bigram_val:>10.4f}")
for name, p, tr, va in results:
    delta = f"{va - prev:+.3f}" if prev is not None else ""
    lines.append(f"{name:<30}{p:>10}{tr:>10.4f}{va:>10.4f}  {delta}")
    prev = va
lines.append("-" * 66)
lines.append("Δval: bir önceki satıra göre değişim (negatif = iyileşme)")
table = "\n".join(lines)
print("\n" + table)

try:
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write(table + "\n")
    print(f"tablo kaydedildi: {RESULTS_PATH}")
except OSError as e:
    print("tablo dosyaya yazılamadı:", e)

# ---------------------------------------------------------------------------
# Son modelden (tam blok) örnek metin
# ---------------------------------------------------------------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nÜretilen metin (3 blok + residual + LayerNorm):\n")
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))


# ###########################################################################
#   LayerNorm vs BatchNorm1d (Hafta 6)
#
#   Videoda  Karpathy, makemore'daki BatchNorm1d'yi alıp TEK bir
#   satırı değiştirerek LayerNorm1d'ye çeviriyor: mean/var'ın alındığı
#   eksen 0 -> 1. Sonra running_mean / running_var ve training bayrağı
#   gereksiz kalıyor ve siliniyor. Aşağıda ikisi yan yana.
# ###########################################################################
print("\n" + "=" * 70)
print("LayerNorm vs BatchNorm1d")
print("=" * 70)

class BatchNorm1d:   # Hafta 6 / makemore part 3

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta  = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var  = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)     # eksen 0: BATCH boyunca (her özellik için)
            xvar  = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar  = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * xvar
        return self.out

class LayerNorm1d:   # videodaki sürüm

    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta  = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)                   # eksen 1: ÖZELLİKLER boyunca (her örnek için)
        xvar  = x.var(1, keepdim=True, unbiased=False)    # nn.LayerNorm ile aynı (yanlı) varyans
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

torch.manual_seed(1337)
x = torch.randn(32, 100) * 3 + 5      # (batch, özellik): ortalama ~5, std ~3
bn = BatchNorm1d(100)
ln = LayerNorm1d(100)
xb = bn(x)
xl = ln(x)

# --- (1) Hangi eksende normalize ediyor? ---
print("\n(1) Hangi eksen?  x.shape =", tuple(x.shape), "= (batch, özellik)")
print(f"{'':<12}{'sütun 0 (1 özellik, tüm batch)':>34}{'satır 0 (1 örnek, tüm özellikler)':>36}")
print(f"{'girdi':<12}{f'mean {x[:, 0].mean():6.3f}  std {x[:, 0].std():5.3f}':>34}"
      f"{f'mean {x[0, :].mean():6.3f}  std {x[0, :].std():5.3f}':>36}")
print(f"{'BatchNorm':<12}{f'mean {xb[:, 0].mean():6.3f}  std {xb[:, 0].std():5.3f}':>34}"
      f"{f'mean {xb[0, :].mean():6.3f}  std {xb[0, :].std():5.3f}':>36}")
print(f"{'LayerNorm':<12}{f'mean {xl[:, 0].mean():6.3f}  std {xl[:, 0].std():5.3f}':>34}"
      f"{f'mean {xl[0, :].mean():6.3f}  std {xl[0, :].std():5.3f}':>36}")
print("-> BatchNorm SÜTUNLARI (0, 1), LayerNorm SATIRLARI (0, 1) yapıyor.")

# --- (2) Örneğin çıktısı batch'teki DİĞER örneklere bağlı mı? ---
x_other = torch.cat([x[:1], torch.randn(31, 100) * 10 - 2])   # aynı 0. örnek, bambaşka komşular
print("\n(2) Aynı örnek, farklı batch içinde -> çıktısı aynı mı?")
print("  BatchNorm:", torch.allclose(bn(x)[0], bn(x_other)[0]))
print("  LayerNorm:", torch.allclose(ln(x)[0], ln(x_other)[0]))

# --- (3) Tek örnek (generate'te B=1) ---
print("\n(3) Batch'te TEK örnek varken (generate sırasında B=1):")
bn3 = BatchNorm1d(100)
for _ in range(200):                          # "eğitim": running stats birikiyor
    bn3(torch.randn(32, 100) * 3 + 5)
bn3.training = False
print("  BatchNorm (eval, running stats):", bn3(x[:1])[0, :4], " <- running_mean sayesinde çalışıyor")
bn3.training = True
print("  BatchNorm (training)           :", bn3(x[:1])[0, :4], " <- 1 örnekten varyans çıkmaz (nan)")
print("  LayerNorm                      :", ln(x[:1])[0, :4], " <- kendi 100 özelliğinden, sorun yok")

# --- (4) Kendi LayerNorm1d'miz == nn.LayerNorm ---
ln_torch = nn.LayerNorm(100)
print("\n(4) LayerNorm1d == nn.LayerNorm:",
      torch.allclose(xl, ln_torch(x), atol=1e-5))

# --- (5) Transformer'da (B, T, C) ---
#   nn.LayerNorm(n_embd) son boyutta normalize eder: her token'ın 32'lik
#   vektörü KENDİ içinde mean 0, std 1 olur. Batch'teki diğer diziler ve
#   aynı dizideki diğer token'lar hiç karışmaz.
x3 = torch.randn(4, 8, n_embd) * 3 + 5
y3 = nn.LayerNorm(n_embd)(x3)
print(f"\n(5) (B, T, C) = {tuple(x3.shape)}: token [0, 3] mean {y3[0, 3].mean():.3f}, "
      f"std {y3[0, 3].std(unbiased=False):.3f}")

