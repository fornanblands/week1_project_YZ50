text = open('/Users/mertisler/Desktop/YZ50/turkceverigpt8.txt', 'r').read()
tokens = text.encode("utf-8")
tokens = list(map(int, tokens))

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


tokens = list(map(int, text.encode('utf-8')))
ids = list(tokens)

vocab_size = 276
num_merges = vocab_size - 256
merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]


#modelden cıkacak tokenleri tekrar metne cevirmek icin decode fonksiyonu yazıyoruz.
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

#modele girecek metin icin tensörleştirme 
#sıralama önemli 

def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


ornek = "Merhaba Çiçekler açtı, ağaçlar yeşillendi 🌸🇹🇷 Şimdi öğrenciler ışıkta çalışıyor 😊"
kodlanmis = encode(ornek)
cozulmus = decode(kodlanmis)

print("Metin        :", ornek)
print("Token'lar    :", kodlanmis)
print("Byte sayısı  :", len(ornek.encode("utf-8")))
print("Token sayısı :", len(kodlanmis))
print("Decode       :", cozulmus)
print("Eşit mi?     :", cozulmus == ornek)

print()
print("decode([128])      :", decode([128]))
print("decode([197])      :", decode([197]))
print("decode([197, 159]) :", decode([197, 159]))

try:
    b"".join(vocab[idx] for idx in [128]).decode("utf-8")
except UnicodeDecodeError as e:
    print("errors='strict' ile:", e)