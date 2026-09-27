import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random



words = open('/Users/mertisler/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
# Karıştırma ve bölme BİR KEZ yapılıyor. Böylece üç model de tam olarak
# aynı isimlerle eğitilip aynı isimlerle test ediliyor.


def build_dataset(words, block_size):
    # Görev 1-4'teki fonksiyonun aynısı; tek fark block_size'ın dışarıdan gelmesi.
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
    # Aynı isim bölmesinden, istenen bağlam uzunluğuyla X, Y üret.
    return (build_dataset(words[:n1], block_size),
            build_dataset(words[n1:n2], block_size),
            build_dataset(words[n2:], block_size))


# ============================================================
# KATMAN SINIFLARI (Görev 4 ile aynı; BatchNorm1d düzeltilmiş hali)
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
    # Görev 4'te düzeltilmiş sürüm: son eksen nöron, geri kalan her eksen örnek.
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
                dim = 0        # (B, C)
            elif x.ndim == 3:
                dim = (0, 1)   # (B, T, C)
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
    # Düz MLP için: (B, T, C) -> (B, T*C). Bütün bağlamı TEK seferde düzleştirir.
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []


class FlattenConsecutive:
    # WaveNet için: (B, T, C) -> (B, T//n, C*n). Sadece ardışık n parçayı birleştirir.
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
# MODEL KURUCULAR
# İki mimari var. Hangi boyutlarla kurulacakları dışarıdan veriliyor.
# ============================================================

def build_mlp(block_size, n_embd, n_hidden):
    # Görev 1 (block_size=3) ve Görev 2 (block_size=8) modeli.
    # Bütün bağlam tek Flatten ile tek vektöre eziliyor, tek gizli katman.
    return Sequential([
        Embedding(vocab_size, n_embd),
        Flatten(),
        Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])


def build_wavenet(block_size, n_embd, n_hidden):
    #  8 parça -> 4 -> 2 -> 1, her adımda ikişer birleştirme.
    # block_size = 8 varsayıyor (2^3 = 8 -> 3 blok).
    assert block_size == 8, 'Bu WaveNet 3 blokla tam 8 harfe göre kurulu'
    return Sequential([
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(2), Linear(n_embd * 2,   n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])


# ============================================================
# TABLONUN ÜÇ SATIRI
# Her satır: isim, mimari kurucu, block_size, n_embd, n_hidden
#
# İlk iki satır Görev 1 ve 2'deki boyutlarla (10, 200) kalıyor: ilerleme
# çizgisini görmek için. Sadece WaveNet büyütülüyor (videodaki gibi 24, 128).
# ============================================================
KONFIG = [
    # isim                    kurucu         block  n_embd  n_hidden
    ('bağlam 3 MLP',          build_mlp,     3,     10,     200),
    ('bağlam 8 düz MLP',      build_mlp,     8,     10,     200),
    ('bağlam 8 WaveNet',      build_wavenet, 8,     24,     128),
]

# Görev 1 ve 2'nin dev loss'unu zaten biliyorsan buraya yaz; o modeller tekrar
# eğitilmez (sadece parametre sayısı hesaplanır). Bilmiyorsan None bırak.
ONCEKI_DEV = {
    'bağlam 3 MLP':     None,   # örn. 2.10 -> Görev 1 çıktından
    'bağlam 8 düz MLP': None,   # örn. 2.03 -> Görev 2 çıktından
}


# ============================================================
# EĞİTİM (içi Görev 1-4 ile birebir aynı)
# ============================================================
max_steps = 200000
batch_size = 32

def train_and_eval(model, Xtr, Ytr, Xdev, Ydev, isim):
    parameters = model.parameters()
    for p in parameters:
        p.requires_grad = True
    lossi = []
    torch.manual_seed(1337)   # üç model de aynı batch-seçme rastgeleliğiyle eğitilsin

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
            print(f'[{isim}] {i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())

    for layer in model.layers:
        layer.training = False

    with torch.no_grad():
        train_loss = F.cross_entropy(model(Xtr), Ytr).item()
        dev_loss = F.cross_entropy(model(Xdev), Ydev).item()
    return lossi, train_loss, dev_loss


# ============================================================
# ÜÇ MODELİ KUR, SAY, (GEREKİRSE) EĞİT
# ============================================================
sonuclar = []   # her eleman: (isim, parametre, train_loss, dev_loss, lossi)
modeller = {}

for isim, kurucu, bs, ne, nh in KONFIG:
    torch.manual_seed(42)                       # her model aynı seed'le başlasın
    model = kurucu(bs, ne, nh)
    with torch.no_grad():
        model.layers[-1].weight *= 0.1          # son katman başta az emin olsun
    n_params = sum(p.nelement() for p in model.parameters())
    # Parametre sayısı eğitimden ÖNCE belli; eğitim sadece değerleri değiştirir, adedi değil.

    print(f'\n=== {isim}: block_size={bs}, n_embd={ne}, n_hidden={nh}, parametre={n_params} ===')

    onceki = ONCEKI_DEV.get(isim)
    if onceki is not None:
        # Görev 1/2 sonucu elde varsa tekrar eğitmeye gerek yok.
        sonuclar.append((isim, n_params, None, onceki, None))
        continue

    (Xtr, Ytr), (Xdev, Ydev), _ = get_splits(bs)
    lossi, tr, dv = train_and_eval(model, Xtr, Ytr, Xdev, Ydev, isim)
    sonuclar.append((isim, n_params, tr, dv, lossi))
    modeller[isim] = (model, bs)


# ============================================================
# TABLO: görevin istediği çıktı
# ============================================================
print('\n' + '=' * 52)
print(f'{"model":22s}{"parametre":>12s}{"dev loss":>12s}')
print('-' * 52)
for isim, n_params, tr, dv, _ in sonuclar:
    print(f'{isim:22s}{n_params:>12,d}{dv:>12.4f}')
print('=' * 52)
# Beklenen (video, küçük sapmalar normal):
#   bağlam 3 MLP          12,097    ~2.10
#   bağlam 8 düz MLP      22,097    ~2.03
#   bağlam 8 WaveNet      76,579    ~1.99


# ============================================================
# LOSS EĞRİLERİ (1000'erli ortalama, eğitilen modeller aynı grafikte)
# ============================================================
for isim, _, _, _, lossi in sonuclar:
    if lossi is not None:
        plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1), label=isim)
plt.title('loss (log10): üç model')
plt.legend()
plt.show()


# ============================================================
# ÖRNEK ÜRETİMİ (büyütülmüş WaveNet)
# ============================================================
if 'bağlam 8 WaveNet' in modeller:
    model, bs = modeller['bağlam 8 WaveNet']
    for _ in range(20):
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
        print(''.join(itos[i] for i in out))