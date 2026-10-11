import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# ###########################################################################
#                          HAFTA 8 — GÖREV 3
#   Modeli büyüt ve dropout ekle. Donanıma sığan bir konfigürasyon seç:
#   n_embd, n_head, n_layer, block_size. Parametre sayısı, val loss, eğitim
#   süresi, seçim gerekçesi ve örnek metin.
#   (1:37:49 - 1:42:39)
#
#   ALGORİTMA
#     1. Cihazı seç: cuda > mps (Apple GPU) > cpu
#     2. Veriyi hazırla (Görev 1-2 ile aynı)
#     3. Model: Görev 2'deki tam blok (residual + pre-LayerNorm) + DROPOUT
#     4. Aday konfigürasyonlar: küçükten büyüğe 4 tane (en büyüğü Karpathy'nin)
#     5. Her adayı birkaç adım eğitip adım süresini ölç
#          -> toplam eğitim süresini tahmin et
#          -> zaman bütçesine (TIME_BUDGET_MIN) sığan EN BÜYÜK adayı seç
#     6. Seçilen modeli eğit, süreyi ölç
#     7. Rapor: parametre, train/val loss, süre, ölçüm tablosu, örnek metin
#        -> ekrana ve dosyaya yaz
#
#   Görev 2'ye göre DEĞİŞENLER:
#     - Dropout: attention ağırlıklarında, proj'dan sonra, ffwd'den sonra
#     - Boyutlar büyüdü; boyutlar artık global değil, sınıflara parametre
#       olarak veriliyor (aynı dosyada farklı boyutlu modelleri ölçebilmek için)
#     - Model artık hep tam blok: residual + LayerNorm her zaman açık
#   AYNI KALANLAR: Head/MultiHead/FeedForward/Block'un mantığı, veri,
#     get_batch, loss, eğitim döngüsü, generate.
# ###########################################################################

# ---------------------------------------------------------------------------
# Ayarlar
TIME_BUDGET_MIN = 30      # eğitime ayırabileceğin süre (dakika). Algoritma buna göre seçer.
FORCE_PRESET    = None    # ölçümü atlayıp elle seçmek için: "S", "M", "L" veya "XL"
RUN_NO_DROPOUT  = False   # True: seçilen modeli dropout=0 ile de eğit (süre 2 katına çıkar)

max_iters     = 5000      # Karpathy ile aynı
eval_interval = 500
eval_iters    = 200
dropout       = 0.2       # Karpathy ile aynı

DATA_PATH     = '/Users/mertisler/Desktop/YZ50/Tiny_Shakespeare.py'
BASELINE_PATH = '/Users/mertisler/Desktop/YZ50/baseline.txt'
REPORT_PATH   = '/Users/mertisler/Desktop/YZ50/week8/week8_gorev3_rapor.txt'
# ---------------------------------------------------------------------------

# ===========================================================================
# 1) Cihaz
#    MacBook (Apple Silicon) -> 'mps' = Apple GPU. CPU'dan genelde birkaç kat hızlı.
# ===========================================================================
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print("cihaz:", device)

def sync():
    """GPU işleri asenkron çalışır; süre ölçmeden önce bitmelerini bekle."""
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

def free_memory():
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

torch.manual_seed(1337)

# ===========================================================================
# 2) Veri (Görev 1-2 ile aynı; get_batch artık boyutları parametre alıyor)
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

def get_batch(split, batch_size, block_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, cfg):
    out = {}
    model.eval()          # <- dropout KAPANIR (ölçüm gürültüsüz olsun)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, cfg['batch_size'], cfg['block_size'])
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()         # <- dropout tekrar AÇILIR
    return out

# ===========================================================================
# 3) Model: Görev 2'deki tam blok + DROPOUT 
#
#   DROPOUT nedir? Eğitim sırasında her forward'da rastgele seçilen
#   nöronların (burada %20'sinin) çıktısını SIFIRLAR, kalanları 1/(1-p) ile
#   büyütür. Her adımda ağın farklı bir "alt ağı" eğitilmiş olur.
#
#   NEDEN şimdi? Model büyüyünce (~10M parametre) 1M karakterlik metni
#   EZBERLEYEBİLİR: train loss düşer, val loss düşmez ya da yükselir
#   (overfitting). Dropout tek tek nöronlara güvenmeyi engeller; ağ
#   bilgiyi birden çok yola dağıtmak zorunda kalır -> genelleme artar.
#
#   Eğitimde açık, ölçüm ve üretimde KAPALI olmalı. Bunu model.train() /
#   model.eval() yönetiyor; nn.Dropout bu bayrağa bakıyor.
#
#   Karpathy'nin koyduğu 3 yer:
#     (a) Head içinde softmax'tan sonra: bazı token'ların birbirine
#         bakmasını rastgele engeller
#     (b) MultiHeadAttention'da proj'dan sonra: residual yola dönmeden önce
#     (c) FeedForward'ın sonunda: residual yola dönmeden önce
# ===========================================================================
class Head(nn.Module):

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)                       # YENİ (a)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)                                  # YENİ (a)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)                       # YENİ (b)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))                      # YENİ (b)

class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),                                 # YENİ (c)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Görev 2'deki son hal: residual + pre-LayerNorm."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """Videonun sonundaki model: embedding -> n_layer blok -> ln_f -> lm_head."""

    def __init__(self, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout)
                                       for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def build_model(cfg, p_drop):
    return GPTLanguageModel(cfg['n_embd'], cfg['n_head'], cfg['n_layer'],
                            cfg['block_size'], p_drop).to(device)

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# ===========================================================================
# 4) Aday konfigürasyonlar (küçükten büyüğe)
#
#   Kurallar:
#     - n_embd, n_head'e tam bölünmeli (head_size = n_embd / n_head)
#     - head_size'ı 16-64 aralığında tut (çok küçükse q·k anlamsızlaşır)
#     - parametre ≈ 12 * n_layer * n_embd²  (+ embedding'ler)
#     - hesap ≈ parametre × batch_size × block_size  -> süreyi bu belirler
#     - model büyüdükçe learning rate düşer (Karpathy: 1e-3 -> 3e-4)
#   XL = Karpathy'nin videodaki konfigürasyonu (A100 GPU'da ~15 dk).
# ===========================================================================
presets = [
    dict(name="S",  batch_size=32, block_size=32,  n_embd=64,  n_head=4, n_layer=4, lr=1e-3),
    dict(name="M",  batch_size=32, block_size=64,  n_embd=128, n_head=4, n_layer=4, lr=6e-4),
    dict(name="L",  batch_size=64, block_size=128, n_embd=192, n_head=6, n_layer=6, lr=3e-4),
    dict(name="XL", batch_size=64, block_size=256, n_embd=384, n_head=6, n_layer=6, lr=3e-4),
]
for cfg in presets:
    assert cfg['n_embd'] % cfg['n_head'] == 0

# ===========================================================================
# 5) Ölç ve seç
#
#   Bir eğitim adımının süresini ölç (ilk 2 adım ısınma, sayılmaz).
#   Toplam süre tahmini:
#     eğitim   = adım_süresi × max_iters
#     ölçüm    = (değerlendirme sayısı × 2 split × eval_iters) forward
#                forward ≈ eğitim adımının ~1/3'ü (backward + optimizer yok)
#   Bütçeye sığan en büyük aday seçilir. Bir aday bütçeyi aşınca daha
#   büyükleri de aşacağı için ölçmeye devam edilmez.
# ===========================================================================
def measure_step_time(cfg, n_steps=5, warmup=2):
    torch.manual_seed(1337)
    model = build_model(cfg, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    for i in range(warmup + n_steps):
        if i == warmup:
            sync()
            t0 = time.time()
        xb, yb = get_batch('train', cfg['batch_size'], cfg['block_size'])
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sync()
    dt = (time.time() - t0) / n_steps
    p = n_params(model)
    del model, opt
    free_memory()
    return dt, p

def estimate_total_sec(step_sec):
    n_evals = max_iters // eval_interval + 1
    eval_forwards = n_evals * 2 * eval_iters
    return step_sec * max_iters + (step_sec / 3) * eval_forwards

bench = []        # (isim, parametre, adım ms, tahmini dakika, sığıyor mu)
if FORCE_PRESET is not None:
    chosen = next(c for c in presets if c['name'] == FORCE_PRESET)
    print(f"\nelle seçildi: {FORCE_PRESET} (ölçüm atlandı)")
else:
    print(f"\n--- adayların ölçümü (bütçe: {TIME_BUDGET_MIN} dk) ---")
    chosen = None
    for cfg in presets:
        try:
            step_sec, p = measure_step_time(cfg)
        except RuntimeError as e:          # bellek yetmezse
            print(f"{cfg['name']:<3} sığmadı (bellek): {str(e)[:60]}")
            bench.append((cfg['name'], None, None, None, False))
            break
        est_min = estimate_total_sec(step_sec) / 60
        fits = est_min <= TIME_BUDGET_MIN
        bench.append((cfg['name'], p, step_sec * 1000, est_min, fits))
        print(f"{cfg['name']:<3} {p / 1e6:6.2f}M param  {step_sec * 1000:8.1f} ms/adım  "
              f"tahmini {est_min:6.1f} dk  {'✓' if fits else '✗ bütçeyi aşıyor'}")
        if fits:
            chosen = cfg
        else:
            break
    if chosen is None:
        chosen = presets[0]
        print("hiçbir aday bütçeye sığmadı -> en küçüğü (S) seçildi; max_iters'ı düşürmeyi düşün")

print(f"\nSEÇİLEN: {chosen}")

# ===========================================================================
# 6) Eğitim
# ===========================================================================
def train(cfg, p_drop, label):
    torch.manual_seed(1337)
    model = build_model(cfg, p_drop)
    p = n_params(model)
    print(f"\n--- {label}: {p / 1e6:.2f}M parametre, dropout={p_drop} ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    sync()
    t0 = time.time()
    history = []
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, cfg)
            elapsed = time.time() - t0
            history.append((it, losses['train'].item(), losses['val'].item(), elapsed))
            print(f"step {it:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}"
                  f"   ({elapsed / 60:5.1f} dk)")
        xb, yb = get_batch('train', cfg['batch_size'], cfg['block_size'])
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    sync()
    total = time.time() - t0
    final = estimate_loss(model, cfg)
    return model, dict(params=p, train=final['train'].item(), val=final['val'].item(),
                       minutes=total / 60, history=history)

model, res = train(chosen, dropout, f"model {chosen['name']}")

res_nodrop = None
if RUN_NO_DROPOUT:
    _, res_nodrop = train(chosen, 0.0, f"model {chosen['name']} (dropout yok)")

# ===========================================================================
# 7) Rapor
# ===========================================================================
model.eval()      # üretimde dropout KAPALI
torch.manual_seed(1337)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    sample = decode(model.generate(context, max_new_tokens=1000)[0].tolist())

try:
    with open(BASELINE_PATH) as f:
        bigram_val = float(f.readline().split()[-1])
except (FileNotFoundError, ValueError):
    bigram_val = None

L = []
L.append("=" * 66)
L.append("HAFTA 8 / GÖREV 3 RAPORU")
L.append("=" * 66)
L.append(f"cihaz            : {device}")
L.append(f"konfigürasyon    : {chosen['name']}  "
         f"n_embd={chosen['n_embd']}  n_head={chosen['n_head']}  "
         f"(head_size={chosen['n_embd'] // chosen['n_head']})  n_layer={chosen['n_layer']}")
L.append(f"                   block_size={chosen['block_size']}  batch_size={chosen['batch_size']}  "
         f"lr={chosen['lr']}  dropout={dropout}  max_iters={max_iters}")
L.append(f"parametre sayısı : {res['params']:,}  ({res['params'] / 1e6:.2f}M)")
L.append(f"train loss       : {res['train']:.4f}")
L.append(f"val loss         : {res['val']:.4f}")
L.append(f"val - train farkı: {res['val'] - res['train']:+.4f}   (büyükse ezberleme var)")
L.append(f"eğitim süresi    : {res['minutes']:.1f} dk")
if bigram_val is not None:
    L.append(f"karşılaştırma    : bigram val {bigram_val:.4f}  ->  bu model {res['val']:.4f}")
if res_nodrop is not None:
    L.append(f"dropout yok      : train {res_nodrop['train']:.4f}  val {res_nodrop['val']:.4f}  "
             f"fark {res_nodrop['val'] - res_nodrop['train']:+.4f}")

if bench:
    L.append("")
    L.append("aday ölçümleri (seçim gerekçesi):")
    L.append(f"  {'aday':<5}{'param':>10}{'ms/adım':>10}{'tahmini dk':>12}  bütçe {TIME_BUDGET_MIN} dk")
    for name, p, ms, est, fits in bench:
        if p is None:
            L.append(f"  {name:<5}{'bellek yetmedi':>32}")
        else:
            L.append(f"  {name:<5}{p / 1e6:>9.2f}M{ms:>10.1f}{est:>12.1f}  {'sığıyor' if fits else 'aşıyor'}")

L.append("")
L.append("eğitim eğrisi:")
for it, tr, va, el in res['history']:
    L.append(f"  step {it:5d}  train {tr:.4f}  val {va:.4f}  ({el / 60:.1f} dk)")

L.append("")
L.append("örnek metin (1000 karakter):")
L.append("-" * 66)
L.append(sample)
report = "\n".join(L)

print("\n" + report)
try:
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\nrapor kaydedildi: {REPORT_PATH}")
except OSError as e:
    print("rapor dosyaya yazılamadı:", e)