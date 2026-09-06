
words = open('/Users/mertisler/names.txt', 'r').read().splitlines()
b = {}
for word in words:
    chs = ["."] + list(word) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = ch1 + ch2
        b[bigram] = b.get(bigram, 0) + 1
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
import torch
N = torch.zeros((27, 27), dtype=torch.int32)
for bigram_str, count in b.items():
    ch1, ch2 = bigram_str[0], bigram_str[1]
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] = count
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis('off')
g = torch.Generator().manual_seed(1)
P = N.float()
P = P / P.sum(1, keepdim=True)   # keepdim: row-wise sum, no broadcasting trap
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
