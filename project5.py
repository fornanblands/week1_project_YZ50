raw_lines = open('/Users/mertisler/Downloads/isimlertürkce/isimlerturkce.py', 'r', encoding='utf-8').read().splitlines()
def tr_lower(s):
    s = s.replace('İ', 'i').replace('I', 'ı')
    return s.lower()
words = []
for line in raw_lines:
    line = line.strip()
    if not line:
        continue
    for token in line.split():
        token = tr_lower(token)
        token = ''.join(ch for ch in token if ch.isalpha())
        if token:
            words.append(token)
words = list(dict.fromkeys(words))
print(f"toplam benzersiz isim sayısı: {len(words)}")
print(words[:10])

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
V = len(stoi)
print(f"\nvocab size: {V}")
print(f"chars: {chars}")

b = {}
for word in words:
    chs = ["."] + list(word) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = ch1 + ch2
        b[bigram] = b.get(bigram, 0) + 1
import torch
N = torch.zeros((V, V), dtype=torch.int32)
for bigram_str, count in b.items():
    ch1, ch2 = bigram_str[0], bigram_str[1]
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] = count
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(V):
    for j in range(V):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis('off')

g = torch.Generator().manual_seed(1)
P = N.float()
P = P / P.sum(1, keepdim=True)   
print("\nsayım modelinden örnek isimler:")
for _ in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

P_smooth = (N + 1).float()
P_smooth = P_smooth / P_smooth.sum(1, keepdim=True)
log_likelihood, n = 0.0, 0
log_likelihood_smooth, n_smooth = 0.0, 0
for word in words:
    chs = ["."] + list(word) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        logprob = torch.log(P[ix1, ix2])
        log_likelihood += logprob
        n += 1
        logprob_smooth = torch.log(P_smooth[ix1, ix2])
        log_likelihood_smooth += logprob_smooth
        n_smooth += 1
nll = -log_likelihood
average_nll = nll / n
nll_smooth = -log_likelihood_smooth
average_nll_smooth = nll_smooth / n_smooth
print(f'\nlog_likelihood: {log_likelihood:.4f}')
print(f'nll: {nll:.4f}')
print(f"average nll (no smoothing): {average_nll:.4f}")
print(f"average nll (with smoothing): {average_nll_smooth:.4f}")
unseen = []
for i in range(V):
    for j in range(V):
        if N[i, j] == 0:
            unseen.append(itos[i] + itos[j])
print(f'\ntotal unseen bigram count: {len(unseen)}')
print(unseen)

import torch.nn.functional as F
xs, ys = [], []
for word in words:
    chs = ["."] + list(word) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(f"\ntotal training bigram count: {num}")
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((V, V), generator=g, requires_grad=True)
for k in range(200):
    xenc = F.one_hot(xs, num_classes=V).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    if k % 20 == 0 or k == 199:
        print(f"step {k:3d}: loss = {loss.item():.4f}")
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad
print(f"\nfinal loss: {loss.item():.4f}")
print(f"task 3 smoothed average nll: {average_nll_smooth:.4f}")
print("check if the two are converging")
g = torch.Generator().manual_seed(2147483647)
print("\nsinir ağı modelinden örnek isimler:")
for _ in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=V).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
