text = " Merhaba dünya! 🚀 2026 yılında Yapay Zeka ve Dil Modelleri üzerine çalışırken, Türkçe 'ğ', 'ç', 'ş', 'ö', 'ü', 'İ' gibi harflerin UTF-8 byte dizilimlerini inceledik. Python'da def tokenize(metin): fonksiyonu ile 50257 tokenlik GPT-2 sözlüğünü test ediyoruz. 130000+ Unicode code point karşısında 256 byte tabanı mucizeler yaratıyor! #NLP #Coding 💻🧠"

code_points = [ord(x) for x in text]  #ord(x) ile text içindeki karakterlerin Unicode code point'ini alıyoruz
print("code points:")
print(code_points)


tokens = text.encode("utf-8")          
tokens = list(map(int, tokens))        
print("\nUTF-8 bytes:")
print(tokens)

#Temelde code points'in mekanizması karaktere özel evrensel sayılar atamak
#fakat atanan bu değerler bir dataset için tüm karakterlere özel olduğundan code points'i bir liste olarak düşünürsek milyonlarca alfabesi olan bir yapı olarak düşünebiliriz.

#burda bizim amacımız ilgili sistemin işlevi oldugundan bu çok sayı hem compute hem bağlam yakalama training acısından cok verimsiz.
#Farklı bir  temsil ihtiyacı hissederiz, bunu da 0-256 arası tamsayılara cevirebildigimiz UTF-8 ile yaparız.



print("\nKarakter sayısı:", len(text))
print("Byte sayısı    :", len(tokens))

# --- 3) Karakter karakter tablo ---
print(f"\n{'kar':>4} {'kod nok.':>9} {'hex':>7}  {'utf-8 byte':<12} {'#byte':>5}")
for ch in text:
    b = list(ch.encode("utf-8"))
    print(f"{ch!r:>4} {ord(ch):>9} {hex(ord(ch)):>7}  {str(b):<12} {len(b):>5}")

# --- 4) Hangi harfler 1 byte, hangileri 2 byte? ---
one_byte = sorted({ch for ch in text if len(ch.encode("utf-8")) == 1})
two_byte = sorted({ch for ch in text if len(ch.encode("utf-8")) == 2})
print("\n1 byte :", one_byte)
print("2 byte :", two_byte)

# --- 5) Geri dönüş kontrolü: byte'lardan metni tekrar elde et ---
assert bytes(tokens).decode("utf-8") == text
print("\nDecode kontrolü: OK")