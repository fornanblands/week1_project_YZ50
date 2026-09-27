import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import ast
import os



# ============================================================
BU_KLASOR = os.path.dirname(os.path.abspath(__file__))        # .../YZ50/week6
YZ50_KLASORU = os.path.dirname(BU_KLASOR)                      # .../YZ50
DOSYA = os.path.join(YZ50_KLASORU, 'isimler.py')

if not os.path.isfile(DOSYA):
    # Yedek: tam yol (script başka yere taşınırsa diye)
    DOSYA = '/Users/mertisler/Desktop/YZ50/isimler.py'
if not os.path.isfile(DOSYA):
    raise FileNotFoundError(
        f'isimler.py bulunamadı. Aranan yer: {DOSYA}\n'
        'Dosyanın YZ50 klasöründe olduğundan emin ol.')
print('veri dosyası:', DOSYA)


# ============================================================
# VERİYİ OKU
# Dosyanın uzantısı .py ama içinde isimler var. İki olasılığı da destekliyoruz:
#   a) Her satırda bir isim (düz metin)
#   b) Python listesi: isimler = ['ahmet', 'ayşe', ...]
# ============================================================
def turkce_kucuk(s):
    # Python'un .lower()'ı 'I' harfini 'i' yapar; Türkçe'de 'ı' olmalı.
    return s.replace('İ', 'i').replace('I', 'ı').lower()


def isimleri_oku(yol):
    metin = open(yol, 'r', encoding='utf-8').read()

    if '[' in metin and ']' in metin:
        # (b) Python listesi: ilk '[' ile son ']' arasını al, güvenli şekilde listeye çevir.
        parca = metin[metin.index('['): metin.rindex(']') + 1]
        ham = ast.literal_eval(parca)
    else:
        # (a) Düz metin: her satır bir isim.
        ham = metin.splitlines()

    isimler = []
    for w in ham:
        w = turkce_kucuk(str(w).strip())
        # Sadece harflerden oluşan isimleri al (boş satır, sayı, noktalama elensin).
        if w and w.isalpha():
            isimler.append(w)

    # Tekrarları at, sırayı koru.
    return list(dict.fromkeys(isimler))


words = isimleri_oku(DOSYA)
print('isim sayısı:', len(words))
print('ilk 10 isim:', words[:10])
assert len(words) > 100, 'Çok az isim okundu; dosyanın içeriğini kontrol et.'

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
print('karakterler:', ''.join(chars))
print('vocab_size:', vocab_size)   # İngilizce'de 27'ydi; Türkçe'de ~30 bekleniyor

uzunluklar = [len(w) for w in words]
print(f'ortalama isim uzunluğu: {sum(uzunluklar) / len(uzunluklar):.2f}')
print(f'3 harften uzun isimlerin oranı: {sum(l > 3 for l in uzunluklar) / len(uzunluklar):.1%}')
# Bağlam 3 iken model, bir ismin 4. harfinden itibaren ismin başını GÖREMEZ.

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
egitim_seti = set(words[:n1])


def build_dataset(words, block_size):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def get_splits(block_size):
    return (build_dataset(words[:n1], block_size),
            build_dataset(words[n1:n2], block_size),
            build_dataset(words[n2:], block_size))


# ============================================================
# KATMAN SINIFLARI (Görev 5 ile aynı)
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
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.view(-1)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.view(-1)
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


class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []


class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
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
# MODEL KURUCULAR (Görev 5 ile aynı)
# ============================================================

def build_mlp(block_size, n_embd, n_hidden):
    return Sequential([
        Embedding(vocab_size, n_embd),
        Flatten(),
        Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])


def build_wavenet(block_size, n_embd, n_hidden):
    assert block_size == 8
    return Sequential([
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(2), Linear(n_embd * 2,   n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])


# ============================================================
# MODELLER
# ============================================================
KONFIG = [
    # isim                         kurucu         block  n_embd  n_hidden
    ('TR bağlam 3 MLP',            build_mlp,     3,     10,     200),
    ('TR bağlam 8 düz MLP',        build_mlp,     8,     10,     200),
    ('TR bağlam 8 WaveNet',        build_wavenet, 8,     24,     128),
]

# Hafta 4'teki Türkçe MLP'nin dev loss'unu buraya yaz (görevin istediği karşılaştırma).
HAFTA4_TR_DEV = None   # örn. 2.25

# Süre kısaltmak istersen True yap: sadece WaveNet eğitilir.
SADECE_WAVENET = False


# ============================================================
# EĞİTİM
# ============================================================
max_steps = 200000     # hızlı deneme için 20000 yapabilirsin (1000'in katı olmalı)
batch_size = 32


def train_and_eval(model, Xtr, Ytr, Xdev, Ydev, isim):
    parameters = model.parameters()
    for p in parameters:
        p.requires_grad = True
    lossi = []
    torch.manual_seed(1337)

    for i in range(max_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)

        for p in parameters:
            p.grad = None
        loss.backward()

        lr = 0.1 if i < 0.75 * max_steps else 0.01   # 200k'da 150k ile aynı
        for p in parameters:
            p.data += -lr * p.grad

        if i % 10000 == 0:
            print(f'[{isim}] {i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())

    for layer in model.layers:
        layer.training = False

    with torch.no_grad():
        train_loss = F.cross_entropy(model(Xtr), Ytr).item()
        dev_loss = F.cross_entropy(model(Xdev), Ydev).item()
    return lossi, train_loss, dev_loss


sonuclar = []
modeller = {}

for isim, kurucu, bs, ne, nh in KONFIG:
    if SADECE_WAVENET and kurucu is not build_wavenet:
        continue
    torch.manual_seed(42)
    model = kurucu(bs, ne, nh)
    with torch.no_grad():
        model.layers[-1].weight *= 0.1
    n_params = sum(p.nelement() for p in model.parameters())
    print(f'\n=== {isim}: block_size={bs}, n_embd={ne}, n_hidden={nh}, parametre={n_params} ===')

    (Xtr, Ytr), (Xdev, Ydev), _ = get_splits(bs)
    lossi, tr, dv = train_and_eval(model, Xtr, Ytr, Xdev, Ydev, isim)
    sonuclar.append((isim, n_params, tr, dv, lossi))
    modeller[isim] = (model, bs)


# ============================================================
# TABLO
# ============================================================
print('\n' + '=' * 64)
print(f'{"model":24s}{"parametre":>12s}{"train loss":>14s}{"dev loss":>12s}')
print('-' * 64)
if HAFTA4_TR_DEV is not None:
    print(f'{"hafta 4 TR MLP":24s}{"-":>12s}{"-":>14s}{HAFTA4_TR_DEV:>12.4f}')
for isim, n_params, tr, dv, _ in sonuclar:
    print(f'{isim:24s}{n_params:>12,d}{tr:>14.4f}{dv:>12.4f}')
print('=' * 64)

d = {isim: dv for isim, _, _, dv, _ in sonuclar}
if 'TR bağlam 3 MLP' in d and 'TR bağlam 8 düz MLP' in d:
    print(f'bağlam 3 -> 8 (aynı MLP):        {d["TR bağlam 8 düz MLP"] - d["TR bağlam 3 MLP"]:+.4f}')
if 'TR bağlam 8 düz MLP' in d and 'TR bağlam 8 WaveNet' in d:
    print(f'düz MLP -> WaveNet (bağlam 8):   {d["TR bağlam 8 WaveNet"] - d["TR bağlam 8 düz MLP"]:+.4f}')
if HAFTA4_TR_DEV is not None and 'TR bağlam 8 WaveNet' in d:
    print(f'hafta 4 TR MLP -> WaveNet:       {d["TR bağlam 8 WaveNet"] - HAFTA4_TR_DEV:+.4f}')


# ============================================================
# LOSS EĞRİLERİ
# ============================================================
for isim, _, _, _, lossi in sonuclar:
    plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1), label=isim)
plt.title('loss (log10): Türkçe isimler')
plt.legend()
plt.show()


# ============================================================
# ÖRNEK İSİM ÜRETİMİ (WaveNet) + yeni mi ezber mi?
# ============================================================
model, bs = modeller['TR bağlam 8 WaveNet']
torch.manual_seed(2147483647)
uretilen = []
for _ in range(30):
    out = []
    context = [0] * bs
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    isim = ''.join(itos[i] for i in out[:-1])   # sondaki '.' hariç
    uretilen.append(isim)
    etiket = '(eğitim setinde VAR)' if isim in egitim_seti else '(yeni)'
    print(f'{isim:15s} {etiket}')

yeni_oran = sum(i not in egitim_seti for i in uretilen) / len(uretilen)
print(f'\nyeni isim oranı: {yeni_oran:.0%}')