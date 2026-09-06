
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
P = P / P.sum(1, keepdim=True)
for _ in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
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
print(f'log_likelihood: {log_likelihood:.4f}')
print(f'nll: {nll:.4f}')
print(f"average nll (smoothing'siz): {average_nll:.4f}")
print(f"average nll (smoothing'li) : {average_nll_smooth:.4f}")
unseen = []
for i in range(27):
    for j in range(27):
        if N[i, j] == 0:
            unseen.append(itos[i] + itos[j])
print(f'\nToplam gorulmemis bigram sayisi: {len(unseen)}')
print(unseen)
test_word = "qq"
chs = ["."] + list(test_word) + ["."]
ll_naive, ll_smooth = 0.0, 0.0
for ch1, ch2 in zip(chs, chs[1:]):
    ix1, ix2 = stoi[ch1], stoi[ch2]
    ll_naive += torch.log(P[ix1, ix2])
    ll_smooth += torch.log(P_smooth[ix1, ix2])
print(f'\n"{test_word}" icin nll (smoothing\'siz): {-ll_naive:.4f}')
print(f'"{test_word}" icin nll (smoothing\'li) : {-ll_smooth:.4f}')










