import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


# ============================================================
# Veri hazırlığı (Görev 2 ile aynı)
# ============================================================
words = open('/Users/mertisler/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

block_size = 8

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr,  Ytr  = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte,  Yte  = build_dataset(words[n2:])

print('Xtr:', Xtr.shape, 'Ytr:', Ytr.shape)

torch.manual_seed(42)


# ============================================================
# KATMAN SINIFLARI
# ============================================================

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
    
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    # !!!  3 boyutlu girdide hatalı .
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    # (B, T, C) -> (B, T//n, C*n)
    # Ardışık n tane vektörü uç uca ekler. Tek parça kalırsa o boyutu atar.
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)   # (B, 1, C*n) -> (B, C*n)
        self.out = x
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ============================================================
# ÖN KONTROL: view gerçekten "ikişer ikişer birleştirme" mi yapıyor?
# ============================================================
e = torch.randn(4, 8, 10)  # (B, T, C)
elle = torch.cat([e[:, ::2, :], e[:, 1::2, :]], dim=2)  # çift + tek indeksler yan yana
view_ile = e.view(4, 4, 20)
print('view == elle birleştirme:', torch.equal(elle, view_ile))  # True olmalı


# ============================================================
#   8 harf -> 4 ikili -> 2 dörtlü -> 1 sekizli
#
# Şekiller B = 4 örneklik batch için yazıldı (eğitimde B = 32 olur).
#   B: örnek sayısı, T: bir örnekteki parça sayısı, C: her parçanın vektör uzunluğu
# ============================================================
n_embd = 10
n_hidden = 68   # her gizli Linear'daki nöron sayısı; ~22k parametre için seçildi

# girdi Xb: (4, 8)
#   4 örnek, her örnekte 8 harfin numarası (0-26). Henüz vektör yok, sadece indeks.

model = Sequential([
    Embedding(vocab_size, n_embd),
    # (4, 8, 10)
    #   Her harf numarası (27, 10)'luk tablodan kendi 10'luk vektörüyle değiştirildi.
    #   Sona yeni bir boyut eklendi: 4 örnek x 8 harf x 10 sayı.

    # ---------------- 1. BLOK: harfleri ikiye paketle ----------------
    FlattenConsecutive(2),
    # (4, 4, 20)
    #   view(4, 4, 20): komşu 2 harfin vektörü uç uca eklendi.
    #   Parça sayısı 8 -> 4 (ikililer), her parça 10 + 10 = 20 sayı.
    #   Toplam sayı değişmedi (8x10 = 4x20 = 80), sadece gruplama değişti.

    Linear(n_embd * 2, n_hidden, bias=False),
    # (4, 4, 68)
    #   Ağırlık (20, 68). @ sadece son boyutla çarpar: 4 örnek x 4 ikili = 16 satırın
    #   her biri AYNI matrisle çarpıldı. Her ikili 20 sayıdan 68 nöron çıktısına indi.
    #   Ön boyutlar (4, 4) korundu, sadece son boyut 20 -> 68.

    BatchNorm1d(n_hidden),
    # (4, 4, 68)
    #   Normalleştirme değerleri ölçekler/kaydırır, şekli değiştirmez.
    #   (Görev 4 notu: içeride mean(0) sadece batch eksenini eziyor,
    #    running_mean (68,) yerine (1, 4, 68) oluyor.)

    Tanh(),
    # (4, 4, 68)
    #   Her sayıya tek tek tanh uygulandı, şekil aynı.

    # ---------------- 2. BLOK: ikilileri ikiye paketle ----------------
    FlattenConsecutive(2),
    # (4, 2, 136)
    #   view(4, 2, 136): bu sefer birleşenler harf değil, 1. bloğun 68'lik çıktıları.
    #   Komşu 2 ikili birleşti: parça sayısı 4 -> 2 (dörtlüler), her parça 68 + 68 = 136.

    Linear(n_hidden * 2, n_hidden, bias=False),
    # (4, 2, 68)
    #   Ağırlık (136, 68). 4 örnek x 2 dörtlü = 8 satır aynı matrisle çarpıldı.
    #   Son boyut 136 -> 68, ön boyutlar (4, 2) korundu.

    BatchNorm1d(n_hidden),
    # (4, 2, 68)
    #   Şekil aynı. (Görev 4 notu: running_mean (1, 2, 68) oluyor.)

    Tanh(),
    # (4, 2, 68)
    #   Şekil aynı.

    # ---------------- 3. BLOK: dörtlüleri ikiye paketle ----------------
    FlattenConsecutive(2),
    # (4, 136)
    #   view önce (4, 1, 136) üretti: 2 dörtlü tek bir sekizliye indi.
    #   Ortada 1 kaldığı için squeeze(1) o boyutu attı -> (4, 136).
    #   Artık her örnek için 8 harfin tamamını temsil eden TEK vektör var,
    #   veri tekrar 2 boyutlu (düz MLP'deki gibi).

    Linear(n_hidden * 2, n_hidden, bias=False),
    # (4, 68)
    #   Klasik 2 boyutlu çarpım: (4, 136) @ (136, 68).

    BatchNorm1d(n_hidden),
    # (4, 68)
    #   Şekil aynı. Girdi 2 boyutlu olduğu için mean(0) burada DOĞRU çalışıyor.

    Tanh(),
    # (4, 68)
    #   Şekil aynı.

    # ---------------- ÇIKTI ----------------
    Linear(n_hidden, vocab_size),
    # (4, 27)
    #   (4, 68) @ (68, 27) + bias(27): her örnek için 27 harfin puanı (logit).
    #   cross_entropy tam olarak bu şekli bekliyor.
])

# ÖZET: her blok aynı örüntüyü tekrarlıyor
#   FlattenConsecutive(2): T yarıya iner, C iki katına çıkar
#   Linear:                T aynı kalır, C -> 68
#   BatchNorm, Tanh:       şekil değişmez
#   T'nin yolculuğu: 8 -> 4 -> 2 -> 1 (atılır). 2^3 = 8 olduğu için 3 blok.

with torch.no_grad():
    model.layers[-1].weight *= 0.1

parameters = model.parameters()
n_params = sum(p.nelement() for p in parameters)
print('parametre sayısı:', n_params)  # 22397 bekleniyor
for p in parameters:
    p.requires_grad = True


# ============================================================
# 4 örneklik bir batch ile ileri geçiş, sonra her katmanın .out şekli.
# ============================================================
with torch.no_grad():
    ix = torch.randint(0, Xtr.shape[0], (4,))
    Xb = Xtr[ix]
    print('\ngirdi Xb:', tuple(Xb.shape))            # (4, 8)
    _ = model(Xb)
    for i, layer in enumerate(model.layers):
        print(f'{i:2d} {layer.__class__.__name__:20s}: {tuple(layer.out.shape)}')
    #  0 Embedding           : (4, 8, 10)
    #  1 FlattenConsecutive  : (4, 4, 20)
    #  2 Linear              : (4, 4, 68)
    #  3 BatchNorm1d         : (4, 4, 68)
    #  4 Tanh                : (4, 4, 68)
    #  5 FlattenConsecutive  : (4, 2, 136)
    #  6 Linear              : (4, 2, 68)
    #  7 BatchNorm1d         : (4, 2, 68)
    #  8 Tanh                : (4, 2, 68)
    #  9 FlattenConsecutive  : (4, 136)
    # 10 Linear              : (4, 68)
    # 11 BatchNorm1d         : (4, 68)
    # 12 Tanh                : (4, 68)
    # 13 Linear              : (4, 27)

# Görev 4 için ipucu: BatchNorm'ların running_mean şekline bak.
print('\nBatchNorm running_mean şekilleri:')
for layer in model.layers:
    if isinstance(layer, BatchNorm1d):
        print('  ', tuple(layer.running_mean.shape))
#   (1, 4, 68)   <- olması gereken (68,) ; 4 konum ayrı ayrı normalleşmiş
#   (1, 2, 68)   <- olması gereken (68,)
#   (1, 68)      <- 2 boyutlu girdi, burada sorun yok


# ============================================================
# EĞİTİM DÖNGÜSÜ (Görev 1-2 ile aynı, tek harf değişmedi)
# ============================================================
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 150000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())


# ============================================================
# LOSS EĞRİSİ
# ============================================================
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
plt.title('loss (log10), WaveNet, block_size=8')
plt.show()


# ============================================================
# DEĞERLENDİRME
# ============================================================
for layer in model.layers:
    layer.training = False


@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())
    return loss.item()

train_loss = split_loss('train')
dev_loss = split_loss('val')


# ============================================================
# SONUÇ ÖZETİ: Görev 2 (düz MLP, bağlam 8) ile karşılaştırma
# ============================================================
gorev2_params = 22097
gorev2_train = None   # Görev 2 çıktından doldur
gorev2_dev = None     # Görev 2 çıktından doldur

print('\n===== KARŞILAŞTIRMA (Görev 2 düz MLP -> Görev 3 WaveNet) =====')
print(f'parametre: {gorev2_params} -> {n_params}  ({n_params - gorev2_params:+d})')
if gorev2_dev is not None:
    print(f'train loss: {gorev2_train:.4f} -> {train_loss:.4f}  ({train_loss - gorev2_train:+.4f})')
    print(f'dev loss:   {gorev2_dev:.4f} -> {dev_loss:.4f}  ({dev_loss - gorev2_dev:+.4f})')
else:
    print(f'train loss: {train_loss:.4f}')
    print(f'dev loss:   {dev_loss:.4f}')


# ============================================================
# MODELDEN ÖRNEK ÜRETİMİ
# ============================================================
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))