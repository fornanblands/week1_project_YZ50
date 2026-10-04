import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# hiperparametreler (Karpathy ng-video-lecture / bigram.py ile aynı)
batch_size = 32       # paralelde kaç örnek işlenecek
block_size = 8        # tahmin için maksimum bağlam (context) uzunluğu
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2  # küçük model -> yüksek lr sorun değil
eval_iters = 200      # loss tahmini için kaç batch ortalaması alınacak
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DEMO = True           # veriyi "tanıma" çıktılarını görmek için
TRAIN_BIGRAM = True   # False: bigram eğitimini atla (Görev 2-3'ü hızlı denemek için)
# ---------------------------------------------------------------------------

torch.manual_seed(1337)

# ===========================================================================
# 1) Veriyi oku ve incele
# ===========================================================================
text = open('/Users/mertisler/Desktop/YZ50/Tiny_Shakespeare.py', 'r', encoding='utf-8').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}      # string -> int
itos = {i: ch for i, ch in enumerate(chars)}      # int -> string
encode = lambda s: [stoi[c] for c in s]           # str  -> list[int]
decode = lambda l: ''.join([itos[i] for i in l])  # list[int] -> str


data = torch.tensor(encode(text), dtype=torch.long)
if DEMO:
    print(data.shape, data.dtype)
    print(data[:100])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ===========================================================================
# 4) block_size fikri: tek bir chunk içinde aslında block_size kaynaklı 8 örnek var
#    model tahmin ederken 8 karakteri görür ama algoritma bize baglı. bigram da yapabiliriz
# ===========================================================================
if DEMO:
    x = train_data[:block_size]
    y = train_data[1:block_size + 1]   # hedef = girdinin bir kaydırılmışı
    for t in range(block_size):
        context = x[:t + 1]
        target = y[t]
        print(f"bağlam {context.tolist()} -> hedef {target.item()}")

# ===========================================================================
# 5) get_batch: rastgele konumlardan 32 tane 8 karakter çek
#    x, y: (B, T) = (batch_size, block_size)
# 32 x 8= 256 karakteri paralel işliyoruz bu harften sonra hangi harf gelir(bigram) tahmin etmek için
# ===========================================================================
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # başlangıç indeksleri
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

if DEMO:
    xb, yb = get_batch('train')
    print('inputs:', xb.shape)   # (32, 8)
    print('targets:', yb.shape)  # (32, 8)

# ===========================================================================
# 6) Loss tahmini
#    Tek batch'in loss'u gürültülü. eval_iters batch'in ortalamasını al.
#    no_grad: backward yapmayacağız, grafik tutmaya gerek yok.
#  forward kısmında cross_entroy tanımladık  loss uzerinden
# ===========================================================================
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ===========================================================================
# 7) Bigram modeli
#    Her token, bir sonraki token'ın logit'lerini bir lookup tablosundan
#    doğrudan okur. Tablo: (vocab_size, vocab_size). Satır i = "i'den sonra
#    ne gelir?" skorları. Bağlamın geri kalanına hiç bakmaz — sadece son token.
# ===========================================================================
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx ve targets: (B, T) integer tensor
        logits = self.token_embedding_table(idx)  # (B, T, C), C = vocab_size

        if targets is None:
            loss = None
        else:
            # F.cross_entropy (N, C) bekliyor -> B ve T'yi düzleştir
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T) — mevcut bağlam
        # Not: bigram için tüm geçmişi forward'a vermek israf (sadece son token
        # kullanılıyor). Ama fonksiyonu genel tutuyoruz; transformer'da lazım olacak.
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                  # (B, T, C)
            logits = logits[:, -1, :]              # sadece son zaman adımı -> (B, C)
            probs = F.softmax(logits, dim=-1)      # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) örnekle
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# ---- sanity check: eğitimden önce loss ne olmalı? ----
# Tam uniform tahminde loss = -ln(1/65) ≈ 4.17. Rastgele init tam uniform
# olmadığı için biraz daha yüksek (videoda ~4.8) çıkması normal.
xb, yb = get_batch('train')
_, loss0 = m(xb, yb)
print(f"başlangıç loss: {loss0.item():.4f}  (uniform ≈ {torch.log(torch.tensor(float(vocab_size))).item():.4f})")

# eğitim öncesi üretim: tamamen çöp olmalı
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # 0 = '\n'
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))

# ===========================================================================
# 8) Eğitim döngüsü
# ===========================================================================
if TRAIN_BIGRAM:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ===========================================================================
    # 9) Taban sonucu kaydet + örnek üret
    # ===========================================================================
    final = estimate_loss()
    print(f"\nBIGRAM TABANI -> train {final['train']:.4f} | val {final['val']:.4f}")
    with open('/Users/mertisler/Desktop/YZ50/baseline.txt', 'w') as f:
        f.write(f"bigram val_loss {final['val']:.4f}\n")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# ###########################################################################
#                                  GÖREV 2
#   Geçmişin ortalamasını üç yoldan al: for döngüsü, tril matrisi, softmax.
#   Üçünün aynı sonucu verdiğini torch.allclose ile göster.  
# ###########################################################################

print("\n" + "=" * 70)
print("GÖREV 2: geçmişin ortalaması (bag of words)")
print("=" * 70)

# ===========================================================================
# 10) Isınma: matris çarpımı = ağırlıklı toplam (videodaki küçük a @ b örneği)
# ===========================================================================
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)          # her satırın toplamı 1
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print("a (agirliklar)=\n", a)
print("b (veri)=\n", b)
print("c = a @ b (her satir = b'nin ilk t+1 satirlarinin ortalaması)=\n", c)

# ===========================================================================
# 11) Oyuncak veri: B=4 dizi, her biri T=8 token, her token C=2 boyutlu vektör
#     Hedef: xbow[b, t] = x[b, 0..t] vektörlerinin ortalaması
#     (t. token sadece kendisinden ve geçmişten bilgi alır, gelecekten değil)
# ===========================================================================
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print("x.shape:", x.shape)

# ---------------------------------------------------------------------------
# Versiyon 1: for döngüsü (doğru ama yavaş)
# ---------------------------------------------------------------------------
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]              # (t+1, C): geçmiş + kendisi
        xbow[b, t] = torch.mean(xprev, 0) # (C,): zaman boyunca ortalama

# ---------------------------------------------------------------------------
# Versiyon 2: alt üçgen matrisle çarpım
#   wei satır t = [1/(t+1), ..., 1/(t+1), 0, ..., 0]
#   wei @ x -> her satır geçmişin ortalaması, tek matris çarpımında
#   (T, T) @ (B, T, C) -> PyTorch wei'yi B boyunca broadcast eder -> (B, T, C)
# ---------------------------------------------------------------------------
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x

# ---------------------------------------------------------------------------
# Versiyon 3: softmax ile
#   0'lar = "affinity" skorları (şimdilik herkes herkese eşit ilgili)
#   -inf  = gelecek yasak (exp(-inf) = 0)
#   softmax -> satırlar normalize -> Versiyon 2'deki wei'nin aynısı
#   Self-attention'da bu 0'ların yerine veriden gelen q @ k.T skorları gelecek.
# ---------------------------------------------------------------------------
tril = torch.tril(torch.ones(T, T))
wei3 = torch.zeros((T, T))
wei3 = wei3.masked_fill(tril == 0, float('-inf'))
wei3 = F.softmax(wei3, dim=-1)
xbow3 = wei3 @ x

# ---------------------------------------------------------------------------
# Karşılaştırma
#   atol=1e-6: mean() ve matris çarpımı toplamaları farklı sırayla yapar,
#   float32'de ~1e-7 düzeyinde yuvarlama farkı olabilir. Bu "aynı sonuç" demek.
# ---------------------------------------------------------------------------
print("\nwei (Versiyon 2)=\n", wei)
print("wei3 (Versiyon 3, softmax)=\n", wei3)
print("\nxbow[0]  (döngü)  =\n", xbow[0])
print("xbow2[0] (tril)   =\n", xbow2[0])
print("xbow3[0] (softmax)=\n", xbow3[0])

print("\nV1 == V2:", torch.allclose(xbow, xbow2, atol=1e-6),
      "| max fark:", (xbow - xbow2).abs().max().item())
print("V1 == V3:", torch.allclose(xbow, xbow3, atol=1e-6),
      "| max fark:", (xbow - xbow3).abs().max().item())
print("wei == wei3:", torch.allclose(wei, wei3))


# ###########################################################################
#                                  GÖREV 3
#   Tek bir self-attention head: query, key, value + pozisyon embedding.
#   wei'yi yazdır, bir satırını oku. √head_size ile bölmenin etkisini göster.
#   (58:26 - 1:19:11)
# ###########################################################################

print("\n" + "=" * 70)
print("GÖREV 3: tek bir self-attention head")
print("=" * 70)

def show_wei(w, labels):
    """wei matrisini yazdır (satır = soran token, sütun = bakılan token)."""
    print("token'lar:", labels)
    print(w.detach().round(decimals=2))

def read_row(w, t, labels):
    """t. token geçmişteki her token'a ne kadar bakıyor?"""
    print(f"\n{t}. token ({labels[t]}) şunlara bakıyor:")
    for j in range(t + 1):
        print(f"  {labels[j]}: {w[t, j].item():.2f}")
    en_cok = w[t].argmax().item()
    print(f"  En çok baktığı: {labels[en_cok]}")
# ===========================================================================
# 13) Versiyon 4: self-attention (videodaki oyuncak örnek)
#     Görev 2'deki Versiyon 3'ten tek fark: wei artık sıfırlarla değil,
#     q @ k.T ile VERİDEN hesaplanıyor; çarpılan şey x değil v.
#
#     query (q): "ben ne arıyorum?"
#     key   (k): "bende ne var?"
#     value (v): "bana bakan olursa ona ne vereceğim?"
#     q_t · k_j büyükse -> t. token j. token'a çok bakar.
# ===========================================================================
torch.manual_seed(1337)
B, T, C = 4, 8, 32            # batch, zaman, kanal (embedding boyutu)
x = torch.randn(B, T, C)

head_size = 16
key   = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)      # (B, T, 16)
q = query(x)    # (B, T, 16)
wei = q @ k.transpose(-2, -1) * head_size**-0.5   # (B, T, 16) @ (B, 16, T) -> (B, T, T)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))   # gelecek yasak
wei = F.softmax(wei, dim=-1)                      # her satır toplamı 1

v = value(x)    # (B, T, 16)
out = wei @ v   # (B, T, T) @ (B, T, 16) -> (B, T, 16)
print("out.shape:", out.shape)

print("\nwei[0] (Versiyon 4) — artık satırlar eşit değil, veriye bağlı:")
toy_labels = [str(i) for i in range(T)]
show_wei(wei[0], toy_labels)
read_row(wei[0], T - 1, toy_labels)

# ===========================================================================
# 14) Head sınıfı + pozisyon embedding, gerçek Shakespeare karakterleriyle
#     Not: model EĞİTİLMEDİ, ağırlıklar rastgele. Bu yüzden wei'deki desen
#     şimdilik anlamsız, mekanizmayı görmek için bakıyoruz. Eğitimle
#     (bir sonraki görev) anlamlı hale gelecek.
# ===========================================================================
n_embd = 32

class Head(nn.Module):
    """Tek bir self-attention head."""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril bir parametre değil (öğrenilmiyor) -> buffer olarak kaydet
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.last_wei = None   # yazdırmak için son wei'yi sakla

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, hs)
        q = self.query(x)   # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5           # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.last_wei = wei.detach()
        v = self.value(x)   # (B, T, hs)
        out = wei @ v       # (B, T, hs)
        return out

torch.manual_seed(1337)
token_embedding_table    = nn.Embedding(vocab_size, n_embd)   # "hangi karakter?"
position_embedding_table = nn.Embedding(block_size, n_embd)   # "kaçıncı sırada?"
head = Head(head_size=16)

idx = train_data[:block_size].unsqueeze(0)                     # (1, T): "First Ci"
labels = [repr(itos[i]) for i in idx[0].tolist()]
print("\nörnek dizi:", repr(decode(idx[0].tolist())))

tok_emb = token_embedding_table(idx)                          # (1, T, C)
pos_emb = position_embedding_table(torch.arange(block_size))  # (T, C)
x = tok_emb + pos_emb                                         # (1, T, C) broadcast

# ---- pozisyon embedding neden gerekli? ----
# Attention'ın kendisi sıra bilmez: sadece vektörlere bakar.
# "First Ci" içinde 'i' hem 1. hem 7. pozisyonda.
print("\nPozisyon embedding olmadan 1. ve 7. 'i' aynı vektör mü?",
      torch.allclose(tok_emb[0, 1], tok_emb[0, 7]))
print("Pozisyon embedding eklenince aynı vektör mü?          ",
      torch.allclose(x[0, 1], x[0, 7]))

out = head(x)
print("\nhead çıktısı:", out.shape)                            # (1, 8, 16)
print("\nwei (gerçek karakterler, eğitilmemiş head):")
show_wei(head.last_wei[0], labels)
read_row(head.last_wei[0], block_size - 1, labels)

# ===========================================================================
# 15) Neden √head_size'a bölüyoruz? (scaled attention)
#     q ve k'nin elemanları ~N(0,1) ise q·k, head_size tane çarpımın toplamı
#     -> varyansı ≈ head_size. Skorlar büyüdükçe softmax "sivrilir":
#     en büyük skora neredeyse 1, diğerlerine neredeyse 0 verir (one-hot).
#     O zaman her token başlangıçta tek bir token'a bakar ve gradyanlar
#     neredeyse sıfır olur. √head_size'a bölmek varyansı tekrar ≈ 1 yapar.
# ===========================================================================
torch.manual_seed(1337)
B, T, head_size = 4, 8, 16
k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)
wei_raw    = q @ k.transpose(-2, -1)          # bölmeden
wei_scaled = wei_raw * head_size**-0.5        # bölerek

print(f"\nvaryanslar (head_size = {head_size}):")
print(f"  k.var()          = {k.var().item():.3f}")
print(f"  q.var()          = {q.var().item():.3f}")
print(f"  (q@k.T).var()    = {wei_raw.var().item():.3f}   <- ≈ head_size")
print(f"  (q@k.T/√hs).var()= {wei_scaled.var().item():.3f}   <- ≈ 1")

# Son satır (t = T-1) maskeden etkilenmez, 8 token'ın hepsini görür.
row_raw    = wei_raw[0, -1]
row_scaled = wei_scaled[0, -1]
p_raw      = F.softmax(row_raw, dim=-1)
p_scaled   = F.softmax(row_scaled, dim=-1)

print("\nSon token'ın softmax çıktısı, yan yana:")
print(f"{'j':>3} {'skor (ham)':>12} {'skor (/√hs)':>12} {'softmax ham':>13} {'softmax /√hs':>13}")
for j in range(T):
    print(f"{j:>3} {row_raw[j].item():12.3f} {row_scaled[j].item():12.3f} "
          f"{p_raw[j].item():13.3f} {p_scaled[j].item():13.3f}")
print(f"en büyük olasılık:  ham = {p_raw.max().item():.3f}   ölçekli = {p_scaled.max().item():.3f}")

# Videodaki mini örnek: aynı skorlar 8 ile çarpılınca softmax one-hot'a kayar
s = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])
print("\nsoftmax(s)     =", F.softmax(s, dim=-1))
print("softmax(s * 8) =", F.softmax(s * 8, dim=-1), " <- en büyüğe yığılıyor")

# head_size büyüdükçe sorun büyür (maske yok, sadece softmax davranışı)
print("\nhead_size büyüdükçe: ortalama 'en büyük olasılık' (1/8 = 0.125 tamamen eşit demek)")
print(f"{'head_size':>10} {'var ham':>9} {'var /√hs':>9} {'max p ham':>10} {'max p /√hs':>11}")
for hs in [4, 16, 64, 256]:
    kk = torch.randn(1000, T, hs)
    qq = torch.randn(1000, T, hs)
    raw = qq @ kk.transpose(-2, -1)
    sc  = raw * hs**-0.5
    mp_raw = F.softmax(raw, dim=-1).max(dim=-1).values.mean().item()
    mp_sc  = F.softmax(sc,  dim=-1).max(dim=-1).values.mean().item()
    print(f"{hs:>10} {raw.var().item():9.1f} {sc.var().item():9.2f} {mp_raw:10.3f} {mp_sc:11.3f}")

# ###########################################################################
#                                  GÖREV 4
#   Head'i bigram modelinin içine koy ve eğit. Val loss'u Görev 1'deki
# ###########################################################################

print("\n" + "=" * 70)
print("GÖREV 4: bigram + tek self-attention head")
print("=" * 70)

# ---------------------------------------------------------------------------
# Görev 4 hiperparametreleri (videodaki değerler)
#   Attention yüksek learning rate'e bigram kadar dayanıklı değil -> 1e-3.
#   lr düşük olduğu için daha fazla adım -> 5000.
# ---------------------------------------------------------------------------
max_iters_sa     = 5000
eval_interval_sa = 500
learning_rate_sa = 1e-3
n_embd           = 32     # embedding boyutu (Görev 3'teki ile aynı)

# ===========================================================================
# 16) Model: bigram iskeleti + pozisyon embedding + tek head + lm_head
#
#   Görev 1'deki bigram'a göre AYNI kalanlar:
#     - forward(idx, targets) -> (logits, loss), loss hesabı birebir aynı
#     - generate'in genel yapısı
#   DEĞİŞENLER:
#     - token embedding artık (vocab, vocab) değil (vocab, n_embd)
#     - pozisyon embedding eklendi: x = tok_emb + pos_emb
#     - x önce Head'den geçiyor: token'lar burada birbiriyle konuşuyor
#     - lm_head: n_embd boyutlu vektörü vocab_size logit'e çeviriyor
#     - generate: girdi son block_size karaktere kırpılıyor
# ===========================================================================
class SingleHeadLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)  # hangi harf?
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # kaçıncı sırada?
        self.sa_head = Head(n_embd)                    # tek head, head_size = n_embd = 32
        self.lm_head = nn.Linear(n_embd, vocab_size)   # (32) -> (65) logit

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                 # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb        # (B, T, n_embd)
        x = self.sa_head(x)          # (B, T, n_embd)  <- geçmişten bilgi topla
        logits = self.lm_head(x)     # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # KIRPMA: pozisyon tablosunda sadece block_size satır var
            # ve tril maskesi block_size x block_size. Daha uzun girdi -> hata.
            idx_cond = idx[:, -block_size:]                # (B, ≤block_size)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                      # son pozisyon -> (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

torch.manual_seed(1337)
model = SingleHeadLanguageModel()   # estimate_loss() global 'model'i kullanıyor
model = model.to(device)
print("parametre sayısı:", sum(p.numel() for p in model.parameters()))

# ---------------------------------------------------------------------------
# Kırpma neden zorunlu? Kırpmadan block_size+1 uzunlukta girdi verelim:
# ---------------------------------------------------------------------------
too_long = torch.zeros((1, block_size + 1), dtype=torch.long, device=device)
try:
    model(too_long)
except Exception as e:
    print(f"\nKırpmadan {block_size + 1} karakter verince hata: {type(e).__name__}: {e}")
    print("-> position_embedding_table'da", block_size, "satır var; 9. pozisyonun embedding'i yok.")

# ===========================================================================
# 17) Eğitim (Görev 1'deki döngünün aynısı)
# ===========================================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_sa)

for iter in range(max_iters_sa):
    if iter % eval_interval_sa == 0 or iter == max_iters_sa - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

final_sa = estimate_loss()

# ===========================================================================
# 18) Bigram tabanıyla yan yana
#     TRAIN_BIGRAM = False ise taban baseline.txt'den okunur.
# ===========================================================================
if TRAIN_BIGRAM:
    bigram_val = final['val'].item()
else:
    with open('/Users/mertisler/Desktop/YZ50/baseline.txt') as f:
        bigram_val = float(f.read().split()[-1])     # "bigram val_loss 2.4xxx"

head_val = final_sa['val'].item()
fark = bigram_val - head_val

print("\n" + "-" * 50)
print(f"{'model':<28}{'val loss':>10}")
print("-" * 50)
print(f"{'Bigram (Görev 1)':<28}{bigram_val:>10.4f}")
print(f"{'Bigram + 1 head (Görev 4)':<28}{head_val:>10.4f}")
print("-" * 50)
print(f"düşüş: {fark:.4f}  ({100 * fark / bigram_val:.1f}%)")

# ---------------------------------------------------------------------------
# Eğitilmiş head'in "First Ci" için wei'si (Görev 3'te rastgeleydi)
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    model(idx.to(device))                 # idx: Görev 3'teki "First Ci"
model.train()
print("\nEğitilmiş head, 'First Ci' için wei:")
show_wei(model.sa_head.last_wei[0].cpu(), labels)
read_row(model.sa_head.last_wei[0].cpu(), block_size - 1, labels)

# ---------------------------------------------------------------------------
# Örnek metin (bigram'ın çıktısıyla karşılaştır)
# ---------------------------------------------------------------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nÜretilen metin:\n")
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

