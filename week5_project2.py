import torch
import torch.nn.functional as F
import random

words = open('/Users/mertisler/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

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

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr,  Ytr  = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte,  Yte  = build_dataset(words[n2:])

n_embd = 10
n_hidden = 200

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / ((n_embd * block_size) ** 0.5)
b1 = torch.randn(n_hidden, generator=g) * 0.1
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1
bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden), generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(f"toplam parametre sayisi: {sum(p.nelement() for p in parameters)}")
for p in parameters:
    p.requires_grad = True

batch_size = 32
n = batch_size
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)

hprebn = embcat @ W1 + b1

bnmeani = 1 / n * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff ** 2
bnvar = 1 / (n - 1) * (bndiff2).sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

h = torch.tanh(hpreact)

logits = h @ W2 + b2

logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum ** -1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
          bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
          embcat, emb]:
    t.retain_grad()
for p in parameters:
    p.grad = None
loss.backward()

print(f"\nloss = {loss.item():.4f}\n")

def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad, atol=1e-5)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | tam esit: {str(ex):5s} | yaklasik esit: {str(app):5s} | max fark: {maxdiff}')

print("=== A) CROSS ENTROPY zincirinin geri turevleri ===\n")

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0 / n
cmp('logprobs', dlogprobs, logprobs)

dprobs = (1.0 / probs) * dlogprobs
cmp('probs', dprobs, probs)

dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)

dcounts_sum = -counts_sum ** -2 * dcounts_sum_inv
cmp('counts_sum', dcounts_sum, counts_sum)

dcounts = counts_sum_inv * dprobs
dcounts += torch.ones_like(counts) * dcounts_sum
cmp('counts', dcounts, counts)

dnorm_logits = counts * dcounts
cmp('norm_logits', dnorm_logits, norm_logits)

dlogits = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]).float() * dlogit_maxes
cmp('logits', dlogits, logits)

print("\n=== B) LINEAR KATMAN 2 (logits = h @ W2 + b2) ===\n")

dh = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)
cmp('h', dh, h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)

print("\n=== C) TANH AKTIVASYONU (h = tanh(hpreact)) ===\n")

dhpreact = (1.0 - h ** 2) * dh
cmp('hpreact', dhpreact, hpreact)

print("\n=== D) BATCHNORM zincirinin geri turevleri ===\n")

dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnraw = bngain * dhpreact
dbnbias = dhpreact.sum(0, keepdim=True)
cmp('bngain', dbngain, bngain)
cmp('bnraw', dbnraw, bnraw)
cmp('bnbias', dbnbias, bnbias)

dbndiff = bnvar_inv * dbnraw
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

dbnvar = (-0.5 * (bnvar + 1e-5) ** -1.5) * dbnvar_inv
cmp('bnvar', dbnvar, bnvar)

dbndiff2 = (1.0 / (n - 1)) * torch.ones_like(bndiff2) * dbnvar
cmp('bndiff2', dbndiff2, bndiff2)

dbndiff += 2 * bndiff * dbndiff2
cmp('bndiff', dbndiff, bndiff)

dhprebn = dbndiff.clone()
dbnmeani = (-dbndiff).sum(0, keepdim=True)
cmp('bnmeani', dbnmeani, bnmeani)

dhprebn += 1.0 / n * (torch.ones_like(hprebn) * dbnmeani)
cmp('hprebn', dhprebn, hprebn)

print("\n=== E) LINEAR KATMAN 1 (hprebn = embcat @ W1 + b1) ===\n")

dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)
cmp('embcat', dembcat, embcat)
cmp('W1', dW1, W1)
cmp('b1', db1, b1)

print("\n=== F) EMBEDDING KATMANI ===\n")

demb = dembcat.view(emb.shape)
cmp('emb', demb, emb)

dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix_karakter = Xb[k, j]
        dC[ix_karakter] += demb[k, j]
cmp('C', dC, C)

print("\nTum karsilastirmalar tamamlandi.")