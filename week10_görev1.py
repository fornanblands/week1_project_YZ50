import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, pipeline, set_seed

# ---------------------------------------------------------------------------
# 1) Ağırlıkları yükle
# ---------------------------------------------------------------------------
# "gpt2" = 124M'lik en küçük model (repo adı: openai-community/gpt2).
# Büyükleri: "gpt2-medium" (350M), "gpt2-large" (774M), "gpt2-xl" (1558M).
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf = model_hf.state_dict()  # {isim: tensor} sözlüğü

# ---------------------------------------------------------------------------
# 2) Her ağırlığın adını ve boyutunu yazdır
# ---------------------------------------------------------------------------
print(f"{'isim':<40} {'boyut':<28} {'eleman sayısı':>14}")
print("-" * 84)
for k, v in sd_hf.items():
    print(f"{k:<40} {str(tuple(v.shape)):<28} {v.numel():>14,}")
# ---------------------------------------------------------------------------
# 3) Toplam parametre sayısı
# ---------------------------------------------------------------------------
# Dikkat: state_dict içindeki her şeyi toplamak 124M vermez, çünkü:
#  (a) lm_head.weight ile transformer.wte.weight AYNI tensördür (weight tying).
#      state_dict ikisini de ayrı anahtar olarak listeler -> iki kez sayılır.
#  (b) transformers sürümüne göre state_dict, öğrenilmeyen buffer'lar da
#      içerebilir (ör. attn.bias = nedensel maske). Bunlar parametre değildir.
toplam_sd = sum(v.numel() for v in sd_hf.values())

# Doğru sayım: model.parameters() paylaşılan tensörleri yalnızca bir kez verir
# ve buffer'ları içermez.
toplam_param = sum(p.numel() for p in model_hf.parameters())

print()
print(f"state_dict'teki her şeyin toplamı : {toplam_sd:,}")
print(f"Gerçek parametre sayısı           : {toplam_param:,}")
print(f"wte ve lm_head aynı bellek mi?     : "
      f"{sd_hf['transformer.wte.weight'].data_ptr() == sd_hf['lm_head.weight'].data_ptr()}")

# Elle doğrulama (config değerleriyle):
cfg = model_hf.config
V, T, C, L = cfg.vocab_size, cfg.n_positions, cfg.n_embd, cfg.n_layer  # 50257, 1024, 768, 12
blok = (
    2 * C                      # ln_1 (weight + bias)
    + C * 3 * C + 3 * C        # attn.c_attn (q, k, v birleşik)
    + C * C + C                # attn.c_proj
    + 2 * C                    # ln_2
    + C * 4 * C + 4 * C        # mlp.c_fc
    + 4 * C * C + C            # mlp.c_proj
)
elle = V * C + T * C + L * blok + 2 * C   # wte + wpe + 12 blok + ln_f
print(f"Elle hesap (V={V}, T={T}, C={C}, L={L}): {elle:,}")

# ---------------------------------------------------------------------------
# 4) Pozisyon embedding'leri (wpe) — videodaki gibi incele
# ---------------------------------------------------------------------------
wpe = sd_hf["transformer.wpe.weight"]   # (1024, 768): her satır bir pozisyon
print()
print("wpe ilk 20 değer:", wpe.view(-1)[:20])

# 4a) Tüm tabloyu görüntü olarak çiz: satır = pozisyon (0..1023), sütun = kanal (0..767)
plt.figure(figsize=(8, 8))
plt.imshow(wpe.numpy(), cmap="gray")
plt.xlabel("embedding kanalı (0-767)")
plt.ylabel("pozisyon (0-1023)")
plt.title("GPT-2 wpe (1024 x 768)")
plt.colorbar()
plt.savefig("wpe_imshow.png", dpi=150, bbox_inches="tight")
plt.show()

# 4b) Birkaç sütunu (kanalı) pozisyona göre çiz — videoda 150, 200, 250
plt.figure(figsize=(10, 4))
for kanal in (150, 200, 250):
    plt.plot(wpe[:, kanal].numpy(), label=f"kanal {kanal}")
plt.xlabel("pozisyon")
plt.ylabel("değer")
plt.title("wpe: tek tek kanalların pozisyona göre değişimi")
plt.legend()
plt.savefig("wpe_kanallar.png", dpi=150, bbox_inches="tight")
plt.show()

# 4c) Karşılaştırma için: bir dikkat katmanının ağırlıklarının bir köşesi
# (videoda h.1.attn.c_attn'ın sol üst 300x300'ü)
plt.figure(figsize=(6, 6))
plt.imshow(sd_hf["transformer.h.1.attn.c_attn.weight"][:300, :300].numpy(), cmap="gray")
plt.title("h.1.attn.c_attn.weight [:300, :300]")
plt.savefig("c_attn_kose.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
generator = pipeline("text-generation", model="gpt2")
set_seed(42)
cikti = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
print()
for i, c in enumerate(cikti):
    print(f"[{i}] {c['generated_text']}")


import torch.nn.functional as F
benzerlik = F.cosine_similarity(wpe[500:501], wpe, dim=1)  # 500. kartı herkesle karşılaştır
plt.plot(benzerlik.numpy()); plt.xlabel("pozisyon"); plt.ylabel("500 ile benzerlik"); plt.show()