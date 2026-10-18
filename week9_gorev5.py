import regex as re

text = open('/Users/mertisler/Desktop/YZ50/turkceverigpt8.txt', 'r').read()
tokens = text.encode("utf-8")
tokens = list(map(int, tokens))

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
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


def render(b):
    return b.decode("utf-8", errors="replace").replace("\n", "\\n")


class BasicTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, ids, vocab_size):
        num_merges = vocab_size - 256
        ids = list(ids)
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def _encode_chunk(self, ids):
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        return self._encode_chunk(list(text.encode("utf-8")))

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")



#normalde ingilizce için kesme işareti sınırlı sayıda kullanılıyor belli kalıplarda
#görev 2 deki yaptığımız yersiz fazlaca boşluklu, kesikli veya emoji içeren metinleri optimize ederken bu sorun oluşturur
#eklemeli dillerde kesme işareti bağlamı koparabilen bir hale geliyor

class RegexTokenizer(BasicTokenizer):
    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        super().__init__()
        self.compiled_pattern = re.compile(pattern)

    def train(self, text, vocab_size):
        num_merges = vocab_size - 256
        chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in chunks]
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        out = []
        for ch in re.findall(self.compiled_pattern, text):
            out.extend(self._encode_chunk(list(ch.encode("utf-8"))))
        return out


vocab_size = 512

basic = BasicTokenizer()
basic.train(tokens, vocab_size)

rgx = RegexTokenizer()
rgx.train(text, vocab_size)

b_tokens = [basic.vocab[i] for i in sorted(basic.merges.values())]
r_tokens = [rgx.vocab[i] for i in sorted(rgx.merges.values())]

#print(f"{'#':>3}  {'Basic':<20} {'Regex (GPT-4)':<20}")
for i in range(min(30, len(b_tokens), len(r_tokens))):
    #print(f"{i+1:>3}  {render(b_tokens[i])!r:<20} {render(r_tokens[i])!r:<20}")

    b_set, r_set = set(b_tokens), set(r_tokens)
#print("ortak:", len(b_set & r_set))
#print("sadece basic:", len(b_set - r_set))
#print("sadece regex:", len(r_set - b_set))

cross = [t for t in b_tokens if len(re.findall(GPT4_SPLIT_PATTERN, t.decode("utf-8", errors="ignore"))) > 1]
#print("basic'te parca sinirini asan:", len(cross))
#print([render(t) for t in cross[:40]])

#print("token sayisi: basic =", len(basic.encode(text)), "regex =", len(rgx.encode(text)))

for w in ["Ankara'dan", "İstanbul'da"]:
    print(w, re.findall(GPT4_SPLIT_PATTERN, w), [render(rgx.vocab[i]) for i in rgx.encode(w)])


import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

tr = open('/Users/mertisler/Desktop/YZ50/paragraf_tr.txt', 'r').read()
en = open('/Users/mertisler/Desktop/YZ50/paragraf_en.txt', 'r').read()


def rapor(ad, metin, encode, decode_tek):
    ids = encode(metin)
    kelime = len(metin.split())
    print(f"\n=== {ad} ===")
    print(f"kelime: {kelime}  token: {len(ids)}  token/kelime: {len(ids)/kelime:.2f}")
    print(f"karakter: {len(metin)}  karakter/token: {len(metin)/len(ids):.2f}")
    print([decode_tek(i) for i in ids[:60]])


rapor("cl100k_base - Ingilizce", en, enc.encode,
      lambda i: enc.decode_single_token_bytes(i).decode("utf-8", errors="replace"))
rapor("cl100k_base - Turkce", tr, enc.encode,
      lambda i: enc.decode_single_token_bytes(i).decode("utf-8", errors="replace"))
rapor("kendi tokenizer - Turkce", tr, rgx.encode,
      lambda i: rgx.vocab[i].decode("utf-8", errors="replace"))
rapor("kendi tokenizer - Ingilizce", en, rgx.encode,
      lambda i: rgx.vocab[i].decode("utf-8", errors="replace"))

print("\n=== kelime kelime (Turkce) ===")
for w in tr.split()[:15]:
    gpt4 = [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in enc.encode(" " + w)]
    ben = [rgx.vocab[i].decode("utf-8", errors="replace") for i in rgx.encode(" " + w)]
    print(f"{w:<20} gpt4: {gpt4}   ben: {ben}")