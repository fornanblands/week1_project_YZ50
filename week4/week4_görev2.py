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

# Loss'u geçen haftaki gibi elle hesapla, sonra F.cross_entropy ile aynı sonucu aldığını göster ve neden onu tercih ettiğ

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

print("elle hesaplanan loss:", loss_manual.item())

loss_builtin = F.cross_entropy(logits, Y)

print("F.cross_entropy ile loss:", loss_builtin.item())
print("aradaki fark:", (loss_manual - loss_builtin).abs().item())