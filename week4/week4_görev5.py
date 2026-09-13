import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

words = open('/Users/mertisler/names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 3

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

X, Y = build_dataset(words)

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)

emb = C[X]

# Görev 1'den Görev 2'ye geçiş

W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)

W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
logits = h @ W2 + b2

counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
N = X.shape[0]
loss_manual = -prob[torch.arange(N), Y].log().mean()
loss_builtin = F.cross_entropy(logits, Y)
print("elle loss:", loss_manual.item(), " | F.cross_entropy loss:", loss_builtin.item())

# Görev 2'den Görev 3'e geçiş

def init_parameters(n_embd=2, n_hidden=100, seed=2147483647):
    gen = torch.Generator().manual_seed(seed)
    C = torch.randn((27, n_embd), generator=gen)
    W1 = torch.randn((block_size * n_embd, n_hidden), generator=gen)
    b1 = torch.randn(n_hidden, generator=gen)
    W2 = torch.randn((n_hidden, 27), generator=gen)
    b2 = torch.randn(27, generator=gen)
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    return parameters

def forward(parameters, Xb, Yb):
    C, W1, b1, W2, b2 = parameters
    emb = C[Xb]
    h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    return logits, loss

@torch.no_grad()
def split_loss(parameters, Xs, Ys):
    _, loss = forward(parameters, Xs, Ys)
    return loss.item()

parameters = init_parameters()
X_kucuk, Y_kucuk = X[:32], Y[:32]

for i in range(300):
    logits, loss = forward(parameters, X_kucuk, Y_kucuk)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1
    for p in parameters:
        p.data -= lr * p.grad

tahmin = logits.max(1).indices

random.seed(42)
words_shuffled = words[:]
random.shuffle(words_shuffled)

n1 = int(0.8 * len(words_shuffled))
n2 = int(0.9 * len(words_shuffled))

Xtr,  Ytr  = build_dataset(words_shuffled[:n1])
Xdev, Ydev = build_dataset(words_shuffled[n1:n2])
Xte,  Yte  = build_dataset(words_shuffled[n2:])

parameters = init_parameters()

lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre

lri, lossi = [], []
g_batch = torch.Generator().manual_seed(2147483647)

for i in range(1000):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g_batch)
    logits, loss = forward(parameters, Xtr[ix], Ytr[ix])

    for p in parameters:
        p.grad = None
    loss.backward()
    lr = lrs[i]
    for p in parameters:
        p.data -= lr * p.grad

    lri.append(lre[i].item())
    lossi.append(loss.item())

plt.figure(figsize=(10, 4))
plt.plot(lri, lossi)
plt.xlabel("log10(learning rate)")
plt.ylabel("loss")
plt.title("Learning rate taraması")
plt.savefig('lr_search.png')
plt.close()

en_iyi_index = int(torch.tensor(lossi).argmin())
en_iyi_exponent = lri[en_iyi_index]
iyi_lr = 10 ** en_iyi_exponent

parameters = init_parameters()

max_steps = 20000
batch_size = 32
lossi_d = []
g_batch = torch.Generator().manual_seed(2147483647)
current_lr = iyi_lr

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g_batch)
    logits, loss = forward(parameters, Xtr[ix], Ytr[ix])

    for p in parameters:
        p.grad = None
    loss.backward()

    if i == int(max_steps * 0.8):
        current_lr = current_lr / 10

    for p in parameters:
        p.data -= current_lr * p.grad

    lossi_d.append(loss.item())

plt.figure(figsize=(10, 4))
plt.plot(lossi_d)
plt.xlabel("adim")
plt.ylabel("loss")
plt.title("Eğitim boyunca minibatch loss")
plt.savefig('train_loss.png')
plt.close()

train_loss = split_loss(parameters, Xtr, Ytr)
dev_loss = split_loss(parameters, Xdev, Ydev)
print(f"[Görev 3 referans model]  train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}")

# Görev 3'ten Görev 4'e geçiş

def train(parameters, Xtr, Ytr, max_steps=20000, batch_size=32,
          lr_start=0.1, seed=2147483647, verbose=False):
    g_batch = torch.Generator().manual_seed(seed)
    lr = lr_start
    lossi = []

    for i in range(max_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g_batch)
        logits, loss = forward(parameters, Xtr[ix], Ytr[ix])

        for p in parameters:
            p.grad = None
        loss.backward()

        if i == int(max_steps * 0.8):
            lr = lr / 10

        for p in parameters:
            p.data -= lr * p.grad

        lossi.append(loss.item())
        if verbose and i % 5000 == 0:
            print(f"    adim {i:6d}/{max_steps}: loss {loss.item():.4f}")

    return parameters, lossi

print("\n=== A) HYPERPARAMETRE TARAMASI ===")

configs = [
    (2,  100),
    (10, 100),
    (10, 200),
    (20, 300),
]

sonuclar = []
for n_embd, n_hidden in configs:
    params_cfg = init_parameters(n_embd=n_embd, n_hidden=n_hidden)
    params_cfg, _ = train(params_cfg, Xtr, Ytr, max_steps=20000)
    train_loss_cfg = split_loss(params_cfg, Xtr, Ytr)
    dev_loss_cfg = split_loss(params_cfg, Xdev, Ydev)
    sonuclar.append((n_embd, n_hidden, train_loss_cfg, dev_loss_cfg))
    print(f"  n_embd={n_embd:3d}  n_hidden={n_hidden:3d}  "
          f"train_loss={train_loss_cfg:.4f}  dev_loss={dev_loss_cfg:.4f}")

en_iyi = min(sonuclar, key=lambda x: x[3])
print(f"\n  en iyi konfigürasyon: n_embd={en_iyi[0]}, n_hidden={en_iyi[1]}, "
      f"dev_loss={en_iyi[3]:.4f}")

print("\n=== B) EMBEDDING GÖRSELLEŞTİRME ===")

parameters_2d = init_parameters(n_embd=2, n_hidden=200)
parameters_2d, _ = train(parameters_2d, Xtr, Ytr, max_steps=20000)

C_2d = parameters_2d[0].detach()

plt.figure(figsize=(8, 8))
plt.scatter(C_2d[:, 0], C_2d[:, 1], s=200, color='lightblue')
for i in range(C_2d.shape[0]):
    plt.text(C_2d[i, 0].item(), C_2d[i, 1].item(), itos[i],
             ha="center", va="center", color="black")
plt.grid(True, which='both')
plt.title("Öğrenilen karakter embedding'leri (2D)")
plt.savefig('embedding_2d.png')
plt.close()

print("\n=== C) MLP'DEN ÖRNEK İSİMLER ===")

def sample_mlp(parameters, n=20, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    C, W1, b1, W2, b2 = parameters
    isimler = []
    for _ in range(n):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(itos[ix])
        isimler.append(''.join(out))
    return isimler

mlp_ornekleri = sample_mlp(parameters, n=20)
print("  ", mlp_ornekleri)

print("\n=== D) BİGRAM KARŞILAŞTIRMASI ===")

N_bigram = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N_bigram[stoi[ch1], stoi[ch2]] += 1

P = (N_bigram + 1).float()
P /= P.sum(1, keepdim=True)

def sample_bigram(n=20, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    isimler = []
    for _ in range(n):
        out = []
        ix = 0
        while True:
            ix = torch.multinomial(P[ix], num_samples=1, generator=g).item()
            if ix == 0:
                break
            out.append(itos[ix])
        isimler.append(''.join(out))
    return isimler

bigram_ornekleri = sample_bigram(n=20)

print("  bigram :", bigram_ornekleri)
print("  MLP    :", mlp_ornekleri)

# Görev 4'ten Görev 5'e geçiş

def init_parameters_naive(n_embd=10, n_hidden=200, seed=2147483647):
    gen = torch.Generator().manual_seed(seed)
    C = torch.randn((27, n_embd), generator=gen)
    W1 = torch.randn((block_size * n_embd, n_hidden), generator=gen)
    b1 = torch.randn(n_hidden, generator=gen)
    W2 = torch.randn((n_hidden, 27), generator=gen)
    b2 = torch.randn(27, generator=gen)
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    return parameters

def init_parameters_kaiming(n_embd=10, n_hidden=200, seed=2147483647):
    gen = torch.Generator().manual_seed(seed)
    fan_in = block_size * n_embd
    gain = 5 / 3

    C = torch.randn((27, n_embd), generator=gen)
    W1 = torch.randn((fan_in, n_hidden), generator=gen) * (gain / fan_in ** 0.5)
    b1 = torch.randn(n_hidden, generator=gen) * 0.01
    W2 = torch.randn((n_hidden, 27), generator=gen) * 0.01
    b2 = torch.zeros(27)

    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    return parameters

def forward_debug(parameters, Xb, Yb):
    C, W1, b1, W2, b2 = parameters
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    h_preact = embcat @ W1 + b1
    h = torch.tanh(h_preact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    return logits, loss, h_preact, h

beklenen_loss = -math.log(1 / 27)
print(f"\nBeklenen (ideal) baslangic loss'u: {beklenen_loss:.4f}")

print("\n=== NAIVE INIT (ölçeksiz) ===")

parameters_naive = init_parameters_naive(n_embd=10, n_hidden=200)
Xb, Yb = Xtr[:32], Ytr[:32]

_, loss_naive, h_preact_naive, h_naive = forward_debug(parameters_naive, Xb, Yb)
print(f"  baslangic loss'u : {loss_naive.item():.4f}  (beklenen {beklenen_loss:.4f})")
print(f"  h_preact std     : {h_preact_naive.std().item():.4f}")
print(f"  |h| > 0.99 orani : {(h_naive.abs() > 0.99).float().mean().item()*100:.1f}%")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(h_preact_naive.detach().view(-1).tolist(), bins=50)
plt.title("h_preact dagilimi (naive init)")
plt.xlabel("değer (tanh'a girmeden önce)")

plt.subplot(1, 2, 2)
plt.hist(h_naive.detach().view(-1).tolist(), bins=50)
plt.title("h = tanh(h_preact) dagilimi (naive init)")
plt.xlabel("değer (-1 ile 1 arasi)")

plt.tight_layout()
plt.savefig('histogram_naive.png')
plt.close()

olu_maskesi_naive = (h_naive.abs() > 0.99)

plt.figure(figsize=(10, 4))
plt.imshow(olu_maskesi_naive.detach(), cmap='gray', interpolation='nearest', aspect='auto')
plt.title("Doymuş (beyaz) hücreler -- naive init")
plt.xlabel("nöron indeksi")
plt.ylabel("örnek (batch içindeki)")
plt.savefig('dead_neurons_naive.png')
plt.close()

tam_olu_naive = olu_maskesi_naive.all(0).sum().item()
print(f"  tamamen ölü nöron sayisi: {tam_olu_naive} / {h_naive.shape[1]}")

print("\n=== KAIMING INIT (ölçekli) ===")

parameters_kaiming = init_parameters_kaiming(n_embd=10, n_hidden=200)

_, loss_kaiming, h_preact_kaiming, h_kaiming = forward_debug(parameters_kaiming, Xb, Yb)
print(f"  baslangic loss'u : {loss_kaiming.item():.4f}  (beklenen {beklenen_loss:.4f})")
print(f"  h_preact std     : {h_preact_kaiming.std().item():.4f}")
print(f"  |h| > 0.99 oranı : {(h_kaiming.abs() > 0.99).float().mean().item()*100:.1f}%")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(h_preact_kaiming.detach().view(-1).tolist(), bins=50)
plt.title("h_preact dagildigini (kaiming init)")
plt.xlabel("değer (tanh'a girmeden önce)")

plt.subplot(1, 2, 2)
plt.hist(h_kaiming.detach().view(-1).tolist(), bins=50)
plt.title("h = tanh(h_preact) dagildigini (kaiming init)")
plt.xlabel("değer (-1 ile 1 arasi)")

plt.tight_layout()
plt.savefig('histogram_kaiming.png')
plt.close()

olu_maskesi_kaiming = (h_kaiming.abs() > 0.99)

plt.figure(figsize=(10, 4))
plt.imshow(olu_maskesi_kaiming.detach(), cmap='gray', interpolation='nearest', aspect='auto')
plt.title("Doymuş (beyaz) hücreler -- kaiming init")
plt.xlabel("nöron indeksi")
plt.ylabel("örnek (batch içindeki)")
plt.savefig('dead_neurons_kaiming.png')
plt.close()

tam_olu_kaiming = olu_maskesi_kaiming.all(0).sum().item()
print(f"  tamamen ölü nöron sayısı: {tam_olu_kaiming} / {h_kaiming.shape[1]}")

print("\n=== KARŞILAŞTIRMA ===")
print(f"  {'':20s} {'naive':>12s} {'kaiming':>12s}")
print(f"  {'başlangıç loss':20s} {loss_naive.item():12.4f} {loss_kaiming.item():12.4f}  (beklenen: {beklenen_loss:.4f})")
print(f"  {'h_preact std':20s} {h_preact_naive.std().item():12.4f} {h_preact_kaiming.std().item():12.4f}")
print(f"  {'|h|>0.99 oranı %':20s} {(h_naive.abs() > 0.99).float().mean().item()*100:12.1f} {(h_kaiming.abs() > 0.99).float().mean().item()*100:12.1f}")
print(f"  {'tam ölü nöron':20s} {tam_olu_naive:12d} {tam_olu_kaiming:12d}")