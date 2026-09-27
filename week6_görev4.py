import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# ============================================================
# Veri hazırlığı (Görev 3 ile aynı)
# ============================================================
words = open('/Users/mertisler/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

block_size = 8

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


# ============================================================
# KATMAN SINIFLARI
# ============================================================

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        # @ sadece SON boyutla çarpar; öndeki boyutlar "batch" gibi davranır.
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1dHatali:
    # ESKİ (HATALI) SÜRÜM: Görev 3'teki BatchNorm1d'nin aynısı.
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
            #   her biri sadece 32 değerden. Olması gereken: 68 ortalama, her biri 128 değerden.
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                # (68,) ile (1, 4, 68) toplanınca broadcasting sonucu (1, 4, 68) çıkar.
                # running_mean'in şekli ilk adımda SESSİZCE bozulur, hata mesajı gelmez.
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class BatchNorm1d:
    # YENİ 
    # Kural: SON eksen nöron eksenidir, geri kalan HER eksen "örnek" sayılır.
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
            # --- DÜZELTME BURADA ---
            if x.ndim == 2:
                dim = 0        # (B, C): sadece batch ekseni örnek
            elif x.ndim == 3:
                dim = (0, 1)   # (B, T, C): hem batch hem konum ekseni örnek
            # x.ndim = tensörün kaç boyutlu olduğu. (32, 68) için 2, (32, 4, 68) için 3.
            #
            # x (32, 4, 68) ise:
            #   mean((0, 1)) -> 0. ve 1. eksen birlikte ezilir -> (1, 1, 68)
            #   Her nöron için 32 örnek x 4 konum = 128 değerin ortalaması.
            #   AYNI nöron sayılıyor. Zaten aynı ağırlıktan çıkıyorlar.
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        # gamma, beta (68,): broadcasting ile her örneğin her konumuna uygulanır.
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


class FlattenConsecutive:
    # (B, T, C) -> (B, T//n, C*n). Tek parça kalırsa ortadaki 1'i atar.
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
# ADIM 1: HATAYI GÖSTER
# Eğitimden önce küçük bir tensörle iki sürümün farkını gözle gör.
# ============================================================
print('=' * 60)
print('HATANIN GÖSTERİMİ')
print('=' * 60)

torch.manual_seed(0)
x = torch.randn(32, 4, 68)
# Her konuma farklı bir kayma ekleyelim: konum 0 ortalaması ~0, konum 3 ortalaması ~3.
# Gerçek modelde de konumlar birbirinden farklı dağılımlara sahip olabilir.
x = x + torch.arange(4).view(1, 4, 1)
# torch.arange(4) = [0, 1, 2, 3]; view(1, 4, 1) ile konum eksenine hizalandı.

print('girdi x:', tuple(x.shape))
print('mean(0)      ->', tuple(x.mean(0, keepdim=True).shape), '  # hatalı: konum başına ayrı')
print('mean((0, 1)) ->', tuple(x.mean((0, 1), keepdim=True).shape), '  # doğru: nöron başına tek')

bn_h = BatchNorm1dHatali(68)
bn_d = BatchNorm1d(68)
out_h = bn_h(x)
out_d = bn_d(x)

# Nöron 0'ın her konumdaki ortalamasına bak.
print('\nnöron 0, konum bazında ortalama (çıktıda):')
print('  hatalı :', [round(v, 2) for v in out_h[:, :, 0].mean(0).tolist()])
print('  doğru  :', [round(v, 2) for v in out_d[:, :, 0].mean(0).tolist()])
# Hatalı sürüm her konumu ayrı ayrı 0'a çeker -> konumlar arasındaki fark SİLİNİR.
#   [0, 0, 0, 0] görürsün: model "konum 3'teki değerler büyüktü" bilgisini kaybetti.
# Doğru sürüm nöronun tüm değerlerini birlikte normalleştirir -> konum farkı KORUNUR.
#   Negatiften pozitife artan değerler görürsün, genel ortalama 0.

print('\nrunning_mean şekli, 1 adım sonra:')
print('  hatalı :', tuple(bn_h.running_mean.shape), '  # olması gereken (68,)')
print('  doğru  :', tuple(bn_d.running_mean.shape))


# ============================================================
# MODEL KURUCU
# Mimari Görev 3 ile aynı. Tek fark: hangi BatchNorm sınıfının kullanılacağı
# parametre olarak veriliyor.
# ============================================================
n_embd = 10
n_hidden = 68

def build_model(BN):
    # BN: BatchNorm1dHatali ya da BatchNorm1d. Sınıfın kendisini değişken gibi
    # verebiliyoruz; BN(n_hidden) yazınca o sınıftan bir katman üretilir.
    torch.manual_seed(42)   # iki model de AYNI başlangıç ağırlıklarıyla başlasın
    model = Sequential([
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(2), Linear(n_embd * 2,   n_hidden, bias=False), BN(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BN(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BN(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])
    with torch.no_grad():
        model.layers[-1].weight *= 0.1
    for p in model.parameters():
        p.requires_grad = True
    return model


# ============================================================
# EĞİTİM + DEĞERLENDİRME
# ============================================================
max_steps = 200000
batch_size = 32

def train_and_eval(model, isim):
    parameters = model.parameters()
    lossi = []
    torch.manual_seed(1337)   # iki eğitim de AYNI batch sırasını görsün

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

    # değerlendirme moduna al
    for layer in model.layers:
        layer.training = False

    with torch.no_grad():
        train_loss = F.cross_entropy(model(Xtr), Ytr).item()
        dev_loss = F.cross_entropy(model(Xdev), Ydev).item()

    # running_mean şekillerini kaydet: hatalı modelde bozuk olacaklar
    rm_shapes = [tuple(l.running_mean.shape) for l in model.layers
                 if hasattr(l, 'running_mean')]
    # hasattr(l, 'running_mean'): "bu katmanda running_mean diye bir şey var mı?"
    # Böylece iki BatchNorm sınıfını da ismine bakmadan yakalıyoruz.

    return lossi, train_loss, dev_loss, rm_shapes


# ============================================================
# ADIM 2: İKİ MODELİ EĞİT
# Uyarı: 200k adım iki kez koşuyor, süre Görev 3'ün iki katı.
# Görev 3'te hatalı sürümün dev loss'unu zaten aldıysan RUN_HATALI = False yap
# ve değeri aşağıdaki HATALI_DEV satırına yaz.
# ============================================================
RUN_HATALI = True
HATALI_TRAIN = None   # RUN_HATALI = False ise Görev 3 çıktından doldur
HATALI_DEV = None

sonuclar = {}

if RUN_HATALI:
    m_h = build_model(BatchNorm1dHatali)
    sonuclar['hatalı'] = train_and_eval(m_h, 'hatalı')
else:
    sonuclar['hatalı'] = (None, HATALI_TRAIN, HATALI_DEV, None)

m_d = build_model(BatchNorm1d)
sonuclar['düzeltilmiş'] = train_and_eval(m_d, 'düzeltilmiş')
model = m_d   # örnek üretimi düzeltilmiş modelle yapılacak


# ============================================================
# ADIM 3: YAN YANA KARŞILAŞTIRMA
# ============================================================
print('\n' + '=' * 60)
print('SONUÇ: BatchNorm düzeltmesi öncesi / sonrası')
print('=' * 60)
print(f'{"":15s}{"train loss":>12s}{"dev loss":>12s}')
for isim in ['hatalı', 'düzeltilmiş']:
    _, tr, dv, _ = sonuclar[isim]
    tr_s = f'{tr:.4f}' if tr is not None else '-'
    dv_s = f'{dv:.4f}' if dv is not None else '-'
    print(f'{isim:15s}{tr_s:>12s}{dv_s:>12s}')

h_dev = sonuclar['hatalı'][2]
d_dev = sonuclar['düzeltilmiş'][2]
if h_dev is not None:
    print(f'\ndev loss farkı: {d_dev - h_dev:+.4f}')
    # Negatif = düzeltme loss'u düşürdü. Videoda fark küçük (~0.01 civarı).
    # Küçük olması normal: model hatayla da eğitilebiliyordu, sadece
    # istatistikleri daha az veriyle ve konumları ayrı ayrı hesaplıyordu.

print('\nrunning_mean şekilleri (eğitim sonrası):')
for isim in ['hatalı', 'düzeltilmiş']:
    rm = sonuclar[isim][3]
    if rm is not None:
        print(f'  {isim:12s}: {rm}')
# hatalı      : [(1, 4, 68), (1, 2, 68), (1, 68)]
# düzeltilmiş : [(68,), (68,), (68,)]


# ============================================================
# LOSS EĞRİLERİ (1000'erli ortalama, aynı grafikte)
# ============================================================
for isim in ['hatalı', 'düzeltilmiş']:
    lossi = sonuclar[isim][0]
    if lossi is not None:
        plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1), label=isim)
plt.title('loss (log10), WaveNet: BatchNorm hatalı vs düzeltilmiş')
plt.legend()
plt.show()


# ============================================================
# MODELDEN ÖRNEK ÜRETİMİ (düzeltilmiş model)
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