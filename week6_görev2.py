import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# ============================================================
# GÖREV 2: Bağlamı 3 harften 8'e çıkar.
# Görev 1'deki koddan TEK fark: block_size = 8
# yalnızca bağlam uzunluğundan gelir.
# ============================================================

# ============================================================
# Veri hazırlığı
# ============================================================
words = open('/Users/mertisler/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

block_size = 8  # <-- GÖREV 2'NİN TEK DEĞİŞİKLİĞİ (Görev 1'de 3'tü)

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

print('Xtr:', Xtr.shape, 'Ytr:', Ytr.shape)  # (N, 8) olmalı

# örnek bağlamlara göz at: her satır 8 harf -> 1 hedef harf
for x, y in zip(Xtr[:15], Ytr[:15]):
    print(''.join(itos[ix.item()] for ix in x), '-->', itos[y.item()])

torch.manual_seed(42)  # seed rng for reproducibility


# ============================================================
# KATMAN SINIFLARI (Görev 1 ile aynı)
# ============================================================

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5  # note: kaiming init
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
        # buffers 
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)   # batch mean
            xvar = x.var(0, keepdim=True)     # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize to unit variance
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


class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
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
# MODEL (Görev 1 ile aynı tanım)
# n_embd * block_size artık 10 * 8 = 80 -> ilk Linear otomatik olarak (80, 200)
# ============================================================
n_embd = 10
n_hidden = 200

model = Sequential([
    Embedding(vocab_size, n_embd),
    Flatten(),
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1  # last layer make less confident

parameters = model.parameters()
n_params = sum(p.nelement() for p in parameters)
print('parametre sayisi:', n_params)  # 22097 bekleniyor (Görev 1'de 12097)
for p in parameters:
    p.requires_grad = True

# her katmanın çıktı şeklini bir kez görmek için küçük bir kontrol
with torch.no_grad():
    _ = model(Xtr[:4])
    for layer in model.layers:
        print(f'{layer.__class__.__name__:12s}: {tuple(layer.out.shape)}')


# ============================================================
# EĞİTİM DÖNGÜSÜ (Görev 1 ile aynı)
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

    lr = 0.1 if i < 150000 else 0.01  # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())


# ============================================================
# LOSS EĞRİSİ (1000'erli ortalama)
# ============================================================
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
plt.title(f'loss (log10), block_size={block_size}')
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
# SONUÇ ÖZETİ
# Görev 1'in değerlerini buraya kendi çıktından yaz.
# ============================================================
gorev1_params = 12097
gorev1_train = None   # örn. 2.06 -> Görev 1 çıktından doldur
gorev1_dev = None     # örn. 2.10 -> Görev 1 çıktından doldur

print('\n===== KARŞILAŞTIRMA =====')
print(f'parametre: {gorev1_params} -> {n_params}  (+{n_params - gorev1_params})')
if gorev1_dev is not None:
    print(f'train loss: {gorev1_train:.4f} -> {train_loss:.4f}  ({train_loss - gorev1_train:+.4f})')
    print(f'dev loss:   {gorev1_dev:.4f} -> {dev_loss:.4f}  ({dev_loss - gorev1_dev:+.4f})')
    print(f'train-dev farkli: {gorev1_dev - gorev1_train:.4f} -> {dev_loss - train_loss:.4f}')
else:
    print(f'train loss: {train_loss:.4f}')
    print(f'dev loss:   {dev_loss:.4f}')
    print('(Görev 1 değerlerini gorev1_train / gorev1_dev satirlarina yazarsan fark da hesaplanır)')


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