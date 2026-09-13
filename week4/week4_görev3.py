import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

words = open('/Users/mertisler/names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
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

def init_parameters(seed=2147483647):
    gen = torch.Generator().manual_seed(seed)
    C = torch.randn((27, 2), generator=gen)
    W1 = torch.randn((6, 100), generator=gen)
    b1 = torch.randn(100, generator=gen)
    W2 = torch.randn((100, 27), generator=gen)
    b2 = torch.randn(27, generator=gen)
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    return parameters

def forward(parameters, Xb, Yb):
    C, W1, b1, W2, b2 = parameters
    emb = C[Xb]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    return logits, loss

print("\n=== A) TEK BATCH'İ OVERFİT ET ===")

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

    if i % 50 == 0:
        print(f"  adim {i:3d}: loss {loss.item():.4f}")

print("  son loss:", loss.item())

tahmin = logits.max(1).indices
print("  tahmin :", tahmin.tolist())
print("  gerçek :", Y_kucuk.tolist())
print("  doğru sayisi:", (tahmin == Y_kucuk).sum().item(), "/", len(Y_kucuk))

print("\n=== B) VERİYİ BÖL ===")

random.seed(42)
words_shuffled = words[:]
random.shuffle(words_shuffled)

n1 = int(0.8 * len(words_shuffled))
n2 = int(0.9 * len(words_shuffled))

Xtr,  Ytr  = build_dataset(words_shuffled[:n1])
Xdev, Ydev = build_dataset(words_shuffled[n1:n2])
Xte,  Yte  = build_dataset(words_shuffled[n2:])

print("  Xtr:", Xtr.shape, " Xdev:", Xdev.shape, " Xte:", Xte.shape)

print("\n=== C) İYİ LEARNING RATE BUL ===")

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
print(f"  en düşük loss'un olduğu nokta: lre={en_iyi_exponent:.3f}  ->  lr={iyi_lr:.4f}")

print("\n=== D) TÜM TRAIN SETİNDE EĞİT ===")

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
        print(f"  [lr decay] adim {i}: lr {current_lr*10:.4f} -> {current_lr:.4f}")

    for p in parameters:
        p.data -= current_lr * p.grad

    lossi_d.append(loss.item())
    if i % 2000 == 0:
        print(f"  adim {i:6d}/{max_steps}: loss {loss.item():.4f}")

plt.figure(figsize=(10, 4))
plt.plot(lossi_d)
plt.xlabel("adim")
plt.ylabel("loss")
plt.title("Eğitim boyunca minibatch loss")
plt.savefig('train_loss.png')
plt.close()

print("\n=== F) DEĞERLENDİRME ===")

@torch.no_grad()
def split_loss(parameters, Xs, Ys):
    _, loss = forward(parameters, Xs, Ys)
    return loss.item()

train_loss = split_loss(parameters, Xtr, Ytr)
dev_loss = split_loss(parameters, Xdev, Ydev)

print("  train_loss:", train_loss)
print("  dev_loss  :", dev_loss)

fark = dev_loss - train_loss
if abs(fark) < 0.05:
    print("  train_loss ≈ dev_loss -> muhtemelen UNDERFITTING")
    print("  öneri: kapasiteyi artır (hidden layer boyutu, embedding boyutu)")
else:
    print(f"  train_loss << dev_loss (fark={fark:.4f}) -> muhtemelen OVERFITTING")
    print("  öneri: regularization ekle veya daha fazla veri topla")