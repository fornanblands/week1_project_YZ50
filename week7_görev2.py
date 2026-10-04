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

# ===========================================================================
# 12) Matris çarpımı neden ağırlıklı ortalamadır? (KENDİ CÜMLELERİNLE YAZ)
# ===========================================================================
# TODO:
#
#
#