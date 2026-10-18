text = open('/Users/mertisler/Desktop/YZ50/turkceverigpt8.txt', 'r').read()
tokens = text.encode("utf-8")
tokens = list(map(int, tokens))


#ikilileri saymak icin bir fonksiyon yazıyoruz
#en son ilk haftalarda bigram modelde sayım yaparken kullanmıştık benzer bir fonksiyonu.
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
print(sorted(((v,k) for k,v in stats.items()), reverse=True)[:20])
top_pair = max(stats, key=stats.get)


#üstte topladığımız ilgilili ikilileri birleştirmemiz lazım Byte pair endocing yazıyoruz.

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


 #yaptığı iterasyonda sık geçen ikilileri secip vocab sozluge eklememiz lazım

for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx) #ids kısaltma 
    merges[pair] = idx

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

print("İlk 20 birleştirme:")
for sira, ((p0, p1), idx) in enumerate(list(merges.items())[:20]):
    print(f"{sira + 1:2d}. {vocab[p0]} + {vocab[p1]} -> {vocab[idx].decode('utf-8', errors='replace')}")

print("Byte sayısı:", len(tokens))
print("Token sayısı:", len(ids))
print(f"Sıkıştırma oranı: {len(tokens) / len(ids):.2f}X")