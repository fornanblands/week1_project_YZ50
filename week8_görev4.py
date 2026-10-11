import re
import math
import time
import unicodedata
from collections import Counter

import torch
import torch.nn as nn
from torch.nn import functional as F

# ###########################################################################
#                          HAFTA 8 — GÖREV 4
#   Aynı modeli Türkçe metinle eğit 
#   - Ürettiği metinden örnek ver.
#   - Türkçe metnin kaç farklı karakteri var, Shakespeare'inkiyle karşılaştır.
#   - İki val loss'u doğrudan karşılaştırabilir misin? Neden?
#
#   ALGORİTMA
#     1. Cihazı seç (cuda > mps > cpu)
#     2. İki metni oku: Tiny Shakespeare + Türkçe
#        Türkçe metni hafifçe temizle (Unicode NFC, görünmez karakterler,
#        wiki işaretleri). Ne temizlendiğini say ve raporla.
#     3. KARAKTER ANALİZİ (eğitimden önce, saniyeler sürer)
#          - boyut (byte ve karakter), farklı karakter sayısı (vocab_size)
#          - sadece Türkçede / sadece Shakespeare'de olan karakterler
#          - nadir karakterler (modelin öğrenemeyeceği kadar az geçenler)
#          - "cetvel" baseline'ları: ln(V), unigram entropisi, bigram entropisi
#     4. MODEL: Görev 3'teki GPT'nin AYNISI (aynı preset, aynı hiperparametre).
#        Görev 3 raporundan hangi preset seçildiyse onu okur.
#     5. Türkçe metinle eğit. Shakespeare val loss'unu Görev 3 raporundan al
#        (bulunamazsa Shakespeare'i de burada aynı ayarlarla eğit).
#     6. Türkçe modelden örnek metin üret.
#     7. KARŞILAŞTIRMA TABLOSU: ham val loss + normalize edilmiş ölçüler
#        (bits/karakter, loss/ln V, unigram ve bigram baseline'a göre kazanç,
#        bits/kelime) -> ekrana ve dosyaya yaz.
#
#   Görev 3'e göre DEĞİŞENLER:
#     - Veri artık bir "Dataset" sözlüğü: birden çok metinle çalışabilmek için
#       stoi/itos/vocab_size/train/val her metne ayrı.
#     - GPTLanguageModel vocab_size'ı parametre olarak alıyor (global değil),
#       çünkü Türkçe ve Shakespeare'in vocab'ı farklı.
#   AYNI KALANLAR: Head, MultiHeadAttention, FeedForward, Block, GPT mimarisi,
#     dropout, eğitim döngüsü, get_batch mantığı, generate.
# ###########################################################################

# ---------------------------------------------------------------------------
# Ayarlar
SH_PATH   = '/Users/mertisler/Desktop/YZ50/Tiny_Shakespeare.py'
TR_PATH   = '/Users/mertisler/Desktop/YZ50/turkceverigpt8.txt'
G3_REPORT = '/Users/mertisler/Desktop/YZ50/week8/week8_gorev3_rapor.txt'
REPORT    = '/Users/mertisler/Desktop/YZ50/week8/week8_gorev4_rapor.txt'

PRESET          = None    # None -> Görev 3 raporundaki preset'i kullan. Elle: "S","M","L","XL"
TRAIN_SH_AGAIN  = False   # True -> Shakespeare'i de burada yeniden eğit (süre 2 katı)
CLEAN_TR        = True    # Türkçe metne hafif temizlik uygula
SAVE_CLEANED    = False   # True -> temizlenmiş metni _temiz.txt olarak kaydet

max_iters     = 5000      # Görev 3 ile aynı
eval_interval = 500
eval_iters    = 200
dropout       = 0.2
# ---------------------------------------------------------------------------

presets = {
    "S":  dict(name="S",  batch_size=32, block_size=32,  n_embd=64,  n_head=4, n_layer=4, lr=1e-3),
    "M":  dict(name="M",  batch_size=32, block_size=64,  n_embd=128, n_head=4, n_layer=4, lr=6e-4),
    "L":  dict(name="L",  batch_size=64, block_size=128, n_embd=192, n_head=6, n_layer=6, lr=3e-4),
    "XL": dict(name="XL", batch_size=64, block_size=256, n_embd=384, n_head=6, n_layer=6, lr=3e-4),
}

# ===========================================================================
# 1) Cihaz
# ===========================================================================
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print("cihaz:", device)

def sync():
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

# ===========================================================================
# 2) Metinleri oku + Türkçe metni temizle
#
#   Neden temizlik? Model karakter düzeyinde çalışıyor: metindeki HER sembol
#   vocab'a girer ve model onu üretmeyi öğrenmeye çalışır. Kopyala-yapıştır
#   sırasında gelen görünmez karakterler veya wiki işaretleri hem vocab'ı
#   şişirir hem de üretilen metne "çöp" olarak sızar.
#
#   Unicode NFC özellikle Türkçe için önemli: "ş", "ğ", "İ" gibi harfler bazen
#   iki parça halinde gelir (harf + birleşen işaret, ör. "s" + "̧"). Ekranda
#   aynı görünür ama model için 2 ayrı karakterdir. NFC bunları tek
#   karaktere birleştirir.
# ===========================================================================
def clean_turkish(t):
    stats = Counter()
    n_comb = sum(1 for ch in t if unicodedata.combining(ch))   # ayrık birleşen işaretler (ör. "̈")
    t = unicodedata.normalize('NFC', t)
    if n_comb:
        stats['NFC: ayrık işaret birleştirildi'] = n_comb

    replacements = {
        '\r': '',            # Windows satır sonu
        '﻿': '',        # BOM
        '​': '', '‌': '', '‍': '',   # sıfır genişlikli boşluklar
        ' ': ' ',       # bölünmez boşluk -> normal boşluk
        ' ': ' ', ' ': ' ',
        '\t': ' ',
    }
    for k, v in replacements.items():
        cnt = t.count(k)
        if cnt:
            stats[f'{k!r} -> {v!r}'] += cnt
            t = t.replace(k, v)

    # Wiki işaretleri: [[Bağlantı|görünen]] -> görünen, [[X]] -> X, {{...}} -> sil, <...> -> sil
    patterns = [
        (r'\[\[[^\]|]*\|([^\]]*)\]\]', r'\1', '[[a|b]] -> b'),
        (r'\[\[([^\]]*)\]\]',          r'\1', '[[a]] -> a'),
        (r'\{\{[^{}]*\}\}',            '',    '{{...}} silindi'),
        (r'<[^>\n]{1,80}>',            '',    '<etiket> silindi'),
    ]
    for pat, rep, label in patterns:
        t, k = re.subn(pat, rep, t)
        if k:
            stats[label] += k

    t, k = re.subn(r'[ ]{2,}', ' ', t)          # çoklu boşluk
    if k: stats['çoklu boşluk'] += k
    t, k = re.subn(r'[ ]+\n', '\n', t)          # satır sonu boşlukları
    if k: stats['satır sonu boşluğu'] += k
    t, k = re.subn(r'\n{3,}', '\n\n', t)        # 2'den fazla boş satır
    if k: stats['fazla boş satır'] += k
    return t.strip() + '\n', stats

sh_text = open(SH_PATH, 'r', encoding='utf-8').read()
tr_raw  = open(TR_PATH, 'r', encoding='utf-8').read()
if CLEAN_TR:
    tr_text, clean_stats = clean_turkish(tr_raw)
    if SAVE_CLEANED:
        with open(TR_PATH.replace('.txt', '_temiz.txt'), 'w', encoding='utf-8') as f:
            f.write(tr_text)
else:
    tr_text, clean_stats = tr_raw, Counter()

# ===========================================================================
# 3) Dataset: her metin için kendi vocab'ı, kendi train/val bölünmesi
#    (Görev 1-3'teki kodun aynısı, sadece bir fonksiyonun içine alındı)
# ===========================================================================
def make_dataset(name, text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return dict(name=name, text=text, chars=chars, vocab_size=len(chars),
                stoi=stoi, itos=itos, train=data[:n], val=data[n:], n_train=n)

SH = make_dataset("Shakespeare", sh_text)
TR = make_dataset("Türkçe", tr_text)

def encode(ds, s):
    return [ds['stoi'][c] for c in s if c in ds['stoi']]

def decode(ds, l):
    return ''.join(ds['itos'][i] for i in l)

# ===========================================================================
# 4) Karakter analizi ve "cetvel" baseline'ları
#
#   Val loss = cross-entropy (nat / karakter). Bu sayının ne kadar iyi
#   olduğu METNE bağlı. O yüzden her metin için 3 referans çizgisi:
#
#   ln(V)  : hiçbir şey bilmeyen model (her karaktere 1/V olasılık).
#            V büyüdükçe bu tavan yükselir. Görev 3'teki başlangıç loss'u bu.
#   H1     : unigram entropisi. Model sadece harf sıklıklarını bilseydi
#            (ör. "a" sık, "j" nadir) ulaşacağı loss. Bağlam yok.
#   H2     : bigram koşullu entropisi. Sadece bir önceki karaktere bakan
#            model (Hafta 7'deki bigram'ın ideal hali). train'den sayılır,
#            val üzerinde ölçülür (add-alpha yumuşatma ile).
#
#   Gerçek modelin val loss'u bu çizgilerin altına ne kadar inerse
#   "bağlamdan o kadar çok şey öğrenmiş" demektir.
# ===========================================================================
def unigram_entropy(text):
    cnt = Counter(text)
    N = len(text)
    return -sum((c / N) * math.log(c / N) for c in cnt.values())

def bigram_cond_entropy(ds, alpha=0.1):
    V = ds['vocab_size']
    tr, va = ds['train'], ds['val']
    # (önceki, sonraki) çiftlerini say: çift -> tek sayı (a*V + b), bincount ile hızlı say
    counts = torch.bincount(tr[:-1] * V + tr[1:], minlength=V * V).view(V, V).float()
    P = (counts + alpha) / (counts + alpha).sum(1, keepdim=True)
    logP = torch.log(P)
    return -logP[va[:-1], va[1:]].mean().item()

def text_stats(ds):
    t = ds['text']
    words = t.split()
    cnt = Counter(t)
    return dict(
        bytes=len(t.encode('utf-8')),
        chars=len(t),
        V=ds['vocab_size'],
        lnV=math.log(ds['vocab_size']),
        H1=unigram_entropy(t),
        H2=bigram_cond_entropy(ds),
        words=len(words),
        chars_per_word=len(t) / max(1, len(words)),   # boşluk dahil, kelime başına karakter
        rare=[(c, k) for c, k in cnt.items() if k < 10],
        top=cnt.most_common(12),
    )

print("\nkarakter analizi yapılıyor...")
S_sh = text_stats(SH)
S_tr = text_stats(TR)

show = lambda c: {'\n': '\\n', ' ': '␣', '\t': '\\t'}.get(c, c)
only_tr = sorted(set(TR['chars']) - set(SH['chars']))
only_sh = sorted(set(SH['chars']) - set(TR['chars']))

# ===========================================================================
# 5) Model: Görev 3'ün AYNISI. Tek fark: vocab_size parametre olarak geliyor.
#    (Türkçe ve Shakespeare'in vocab'ı farklı olduğu için embedding tablosu
#     ve lm_head'in boyutu metne göre değişiyor; gövde birebir aynı.)
# ===========================================================================
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)   # <- vocab'a bağlı
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout)
                                       for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)                     # <- vocab'a bağlı

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# ===========================================================================
# 6) Eğitim (Görev 3'teki döngünün aynısı; hangi metinle çalışacağı parametre)
# ===========================================================================
def get_batch(ds, split, batch_size, block_size):
    data = ds['train'] if split == 'train' else ds['val']
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, ds, cfg):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(ds, split, cfg['batch_size'], cfg['block_size'])
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(ds, cfg):
    torch.manual_seed(1337)
    model = GPTLanguageModel(ds['vocab_size'], cfg['n_embd'], cfg['n_head'],
                             cfg['n_layer'], cfg['block_size'], dropout).to(device)
    p = n_params(model)
    print(f"\n--- {ds['name']}: preset {cfg['name']}, {p / 1e6:.2f}M parametre, "
          f"vocab {ds['vocab_size']} ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    sync(); t0 = time.time()
    history = []
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, ds, cfg)
            el = time.time() - t0
            history.append((it, losses['train'].item(), losses['val'].item(), el))
            print(f"step {it:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}   ({el / 60:5.1f} dk)")
        xb, yb = get_batch(ds, 'train', cfg['batch_size'], cfg['block_size'])
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    sync()
    final = estimate_loss(model, ds, cfg)
    return model, dict(params=p, train=final['train'].item(), val=final['val'].item(),
                       minutes=(time.time() - t0) / 60, history=history)

# ---------------------------------------------------------------------------
# Görev 3 raporundan preset'i ve Shakespeare val loss'unu oku
# ---------------------------------------------------------------------------
g3_preset, g3_val = None, None
try:
    rep3 = open(G3_REPORT, encoding='utf-8').read()
    m = re.search(r'konfigürasyon\s*:\s*(\w+)', rep3)
    g3_preset = m.group(1) if m else None
    m = re.search(r'val loss\s*:\s*([0-9.]+)', rep3)
    g3_val = float(m.group(1)) if m else None
except FileNotFoundError:
    pass

name = PRESET or g3_preset or "M"
cfg = presets[name]
print(f"\npreset: {name}  (kaynak: {'elle' if PRESET else ('Görev 3 raporu' if g3_preset else 'varsayılan')})")

model_tr, res_tr = train(TR, cfg)

if TRAIN_SH_AGAIN or g3_val is None or (g3_preset and g3_preset != name):
    _, res_sh = train(SH, cfg)
    sh_val, sh_src = res_sh['val'], 'bu çalıştırma'
else:
    sh_val, sh_src = g3_val, 'Görev 3 raporu'

# ===========================================================================
# 7) Örnek metin (Türkçe model)
#    İki örnek: (a) boş bağlamdan, (b) Türkçe bir başlangıç cümlesinden.
# ===========================================================================
model_tr.eval()
torch.manual_seed(1337)
with torch.no_grad():
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample_a = decode(TR, model_tr.generate(ctx, 800)[0].tolist())
    prompt = "Bir gün"
    ctx = torch.tensor([encode(TR, prompt)], dtype=torch.long, device=device)
    sample_b = decode(TR, model_tr.generate(ctx, 500)[0].tolist())

# ===========================================================================
# 8) Karşılaştırma
#
#   SORU: İki val loss'u doğrudan karşılaştırabilir miyiz?
#   CEVAP: HAYIR, ham haliyle değil. Sebepler:
#
#   (1) Farklı sınıf sayısı. Loss, V sınıflı bir sınıflandırmanın
#       cross-entropy'si. V büyüdükçe "hiç bilmeyen" modelin loss'u ln(V)
#       de büyür. Aynı loss değeri V=65'te ve V=90'da aynı başarı değildir.
#
#   (2) Farklı metin, farklı belirsizlik. Her metnin kendine özgü
#       öngörülebilirliği var (entropi). Shakespeare'de tekrarlayan isimler,
#       büyük harf + iki nokta ile başlayan konuşmacı satırları gibi kolay
#       tahmin edilen kalıplar çok. Türkçe sondan eklemeli: kelimeler uzun,
#       ekler çok çeşitli; bir sonraki harfin belirsizliği farklı.
#
#   (3) Farklı val kümesi. İkisi de "metnin son %10'u" ama bu iki farklı
#       metnin farklı bölümleri; zorlukları eşit değil.
#
#   (4) Karakter != anlam birimi. Türkçede kelime başına karakter sayısı
#       farklı; aynı bilgiyi taşımak için farklı sayıda karakter gerekiyor.
#       Karakter başına loss, kelime başına loss'a farklı oranda dönüşür.
#
#   Ne yapılabilir? Aynı cetvele vurmak: her loss'u kendi metninin
#   baseline'larına göre okumak (aşağıdaki tablo). Bu bile mükemmel değil
#   ama "model kendi metninde ne kadar yol aldı" sorusuna daha adil cevap.
# ===========================================================================
LN2 = math.log(2)
rows = [
    ("boyut (MB, UTF-8)",          f"{S_sh['bytes'] / 1e6:.2f}",       f"{S_tr['bytes'] / 1e6:.2f}"),
    ("karakter sayısı",            f"{S_sh['chars']:,}",               f"{S_tr['chars']:,}"),
    ("farklı karakter (vocab)",    f"{S_sh['V']}",                     f"{S_tr['V']}"),
    ("kelime başına karakter",     f"{S_sh['chars_per_word']:.2f}",    f"{S_tr['chars_per_word']:.2f}"),
    ("ln(V)  hiç bilmeyen",        f"{S_sh['lnV']:.3f}",               f"{S_tr['lnV']:.3f}"),
    ("H1     sadece sıklık",       f"{S_sh['H1']:.3f}",                f"{S_tr['H1']:.3f}"),
    ("H2     sadece önceki harf",  f"{S_sh['H2']:.3f}",                f"{S_tr['H2']:.3f}"),
    ("VAL LOSS (nat/karakter)",    f"{sh_val:.4f}",                    f"{res_tr['val']:.4f}"),
    ("bits/karakter",              f"{sh_val / LN2:.3f}",              f"{res_tr['val'] / LN2:.3f}"),
    ("loss / ln(V)",               f"{sh_val / S_sh['lnV']:.3f}",      f"{res_tr['val'] / S_tr['lnV']:.3f}"),
    ("H1'e göre kazanç",           f"{1 - sh_val / S_sh['H1']:.1%}",   f"{1 - res_tr['val'] / S_tr['H1']:.1%}"),
    ("H2'ye göre kazanç",          f"{1 - sh_val / S_sh['H2']:.1%}",   f"{1 - res_tr['val'] / S_tr['H2']:.1%}"),
    ("bits/kelime (yaklaşık)",     f"{sh_val / LN2 * S_sh['chars_per_word']:.2f}",
                                   f"{res_tr['val'] / LN2 * S_tr['chars_per_word']:.2f}"),
]

L = []
L.append("=" * 70)
L.append("HAFTA 8 / GÖREV 4 RAPORU — Türkçe metinle eğitim")
L.append("=" * 70)
L.append(f"cihaz: {device}   preset: {name}   (n_embd={cfg['n_embd']}, n_head={cfg['n_head']}, "
         f"n_layer={cfg['n_layer']}, block_size={cfg['block_size']}, lr={cfg['lr']})")
L.append(f"Türkçe model: {res_tr['params']:,} parametre, eğitim {res_tr['minutes']:.1f} dk, "
         f"train {res_tr['train']:.4f}, val {res_tr['val']:.4f}")
L.append(f"Shakespeare val loss kaynağı: {sh_src}")

if clean_stats:
    L.append("")
    L.append("Türkçe metin temizliği:")
    for k, v in clean_stats.items():
        L.append(f"  {k:<28} {v}")

L.append("")
L.append("KARAKTER KARŞILAŞTIRMASI")
L.append(f"  Shakespeare: {S_sh['V']} karakter")
L.append(f"  Türkçe     : {S_tr['V']} karakter")
L.append(f"  sadece Türkçede    ({len(only_tr)}): " + ' '.join(show(c) for c in only_tr))
L.append(f"  sadece Shakespeare ({len(only_sh)}): " + ' '.join(show(c) for c in only_sh))
L.append("  Türkçe en sık 12  : " + '  '.join(f"{show(c)}:{k}" for c, k in S_tr['top']))
L.append("  Shakespeare en sık: " + '  '.join(f"{show(c)}:{k}" for c, k in S_sh['top']))
if S_tr['rare']:
    L.append(f"  Türkçede 10'dan az geçen ({len(S_tr['rare'])}): "
             + ' '.join(f"{show(c)}:{k}" for c, k in sorted(S_tr['rare'], key=lambda x: x[1])))
    L.append("    -> bunları model neredeyse hiç göremez; vocab'ı şişirir. Silmeyi düşün.")

L.append("")
L.append("VAL LOSS KARŞILAŞTIRMASI")
L.append(f"  {'ölçü':<28}{'Shakespeare':>14}{'Türkçe':>14}")
L.append("  " + "-" * 56)
for r in rows:
    L.append(f"  {r[0]:<28}{r[1]:>14}{r[2]:>14}")
L.append("  " + "-" * 56)
L.append("  Ham val loss'lar DOĞRUDAN karşılaştırılamaz: vocab (ln V), metnin kendi")
L.append("  entropisi, val kümesi ve kelime başına karakter sayısı farklı. Satırları")
L.append("  kendi baseline'larına göre oku (kazanç satırları).")

L.append("")
L.append("eğitim eğrisi (Türkçe):")
for it, tr, va, el in res_tr['history']:
    L.append(f"  step {it:5d}  train {tr:.4f}  val {va:.4f}  ({el / 60:.1f} dk)")

L.append("")
L.append("ÖRNEK METİN (a) boş bağlamdan, 800 karakter:")
L.append("-" * 70)
L.append(sample_a)
L.append("")
L.append(f"ÖRNEK METİN (b) '{prompt}' ile başlayarak, 500 karakter:")
L.append("-" * 70)
L.append(sample_b)
report = "\n".join(L)

print("\n" + report)
try:
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\nrapor kaydedildi: {REPORT}")
except OSError as e:
    print("rapor dosyaya yazılamadı:", e)