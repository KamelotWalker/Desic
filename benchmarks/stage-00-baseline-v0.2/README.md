# Aşama 00 — v0.2 temel ölçüm (baseline)

**Amaç:** Bundan sonraki her değişikliğin karşılaştırılacağı başlangıç noktasını ölçmek. Kod, bu aşamada değiştirilmedi; sadece ölçüldü.

- **Veri:** Banking77 — 77 bankacılık niyeti, 10.003 eğitim / 3.080 test
- **Kod:** çevrimiçi sonuçlar commit `fd6c9c7`; nöral koşu, eşdeğer kodla (`1c3e42b` ile aynı benchmark scripti) üretildi
- **Ortam:** Python 3.11, 4 çekirdek Intel Xeon @ 2.80 GHz, **GPU yok**, torch 2.14.1 (CPU), 30.09.2026

## Sonuçlar

### Çevrimiçi öğrenci (lineer + ağaç + bellek + öncül uzmanlar, Hedge karışımı)

| Eğitim | Doğruluk | Log loss | ECE | Süre |
|---|---|---|---|---|
| Sınıf başına 10 örnek (770) | %57,2 | 1,702 | 0,122 | — |
| Sınıf başına 25 örnek (1.925) | %72,8 | 0,981 | 0,019 | — |
| Sınıf başına 50 örnek (3.850) | %82,6 | 0,623 | 0,011 | — |
| **Tüm veri, karışık akış (10.003)** | **%88,5** | **0,427** | **0,013** | 92 sn |
| Tüm veri, **sınıfa göre sıralı akış** | %15,8 | 3,369 | 0,474 | 95 sn |

### Seçici tahmin (model yalnızca yeterince eminken cevap verirse)

| Güven eşiği | Karışık akış: kapsam / doğruluk | Sıralı akış: kapsam / doğruluk |
|---|---|---|
| ≥ %50 | %93,4 / %92,0 | %73,2 / %16,6 |
| ≥ %70 | %85,7 / %95,1 | %36,7 / %19,3 |
| ≥ %90 | %73,3 / %97,8 | %8,3 / %42,8 |

### Yerleşik nöral öğrenci (önceden eğitilmemiş küçük encoder, 6 epoch, CPU'da 814 sn)

| Kurulum | Doğruluk | Log loss | ECE |
|---|---|---|---|
| Nöral öğrenci tek başına | %68,4 | 1,149 | 0,015 |
| Çevrimiçi uzmanlar + nöral (başlangıç ağırlığı %30) | %88,3 | 0,440 | 0,035 |

Karışım, en fazla %90 eminken cevap verirse sorunun %61,2'sine %98,6 doğrulukla cevap veriyor.

## Referans

Yayınlanmış sonuçlarda (Casanueva ve ark., 2020, *Efficient Intent Detection with Dual Sentence Encoders*), tam veriyle ince ayarlı BERT ve ConveRT gibi önceden eğitilmiş encoder'lar yaklaşık **%93** doğruluğa ulaşıyor. Desic'in çevrimiçi öğrencisinin önceden eğitilmiş bir dil bilgisi yok; ~5 puanlık fark beklenen bir fark. Jev ve Laya'nın Banking77 için yayınlanmış bir sonucu bilinmiyor; Laya'nın paylaştığı rakamlar farklı bir benchmark'a (typed-decisions) ait olduğundan doğrudan karşılaştırılamaz.

## Bulgular

1. **Kalibrasyon güçlü.** Karışık akışta ECE 0,013: model "%90 eminim" dediğinde gerçekten ~%90 doğru. Bu, risk bütçesi ve "emin değilsen öğretmene sor" mekanizmasının sağlam bir temel üzerinde durduğu anlamına geliyor.
2. **Seçici tahmin işe yarıyor.** Belirsiz olan ~%14'ü öğretmene veya insana bırakınca kalan %86'da hata oranı %11,5'ten %4,9'a iniyor.
3. **Sıralı akışta çöküş — en önemli zayıflık.** Aynı veri sınıf sınıf gelince doğruluk %88,5'ten %15,8'e düşüyor ve kalibrasyon bozuluyor (ECE 0,474). Model son gördüğü sınıflara kayıyor (catastrophic forgetting) ve sıcaklık kalibrasyonu da son örneklerle bozuluyor. Canlı kullanımda aynı türden art arda gelen geri bildirimler benzer bir etki yaratabilir. "Öğren ama kendini bozma" hedefinin **şu an sağlanmadığının** ölçülmüş kanıtı; sonraki aşamanın ilk hedefi.
4. **Yerleşik nöral öğrenci zayıf.** Önceden eğitilmemiş encoder %68,4'te kaldı ve karışıma eklenince log loss ve ECE'yi biraz kötüleştirdi. Sebep: terfi anında sabit %30 başlangıç ağırlığı ve bu koşuda gölge denetiminin kapalı olması (`shadow_min = 0`). Başlangıç ağırlığının offline test sonuçlarından türetilmesi gerekiyor.
5. **Ölçülemeyenler.** Önceden eğitilmiş backbone (mmBERT, ModernBERT) ile nöral öğrenci bu ortamda ölçülemedi (Hugging Face erişimi ve GPU yok). Gerçek bir LLM öğretmenle uçtan uca ölçüm de yapılmadı.

## Notlar

- Sıralı koşudaki az örnekli satırlar (10/25/50 örnek) sınıf başına örneklem sonrası karıştırılıyor; o satırlar sıralı akışı değil, yalnızca farklı bir örneklemi temsil eder. Sıralı akış etkisini gösteren satır "tüm veri"dir.
- Nöral koşu JSON üretmeyen önceki script sürümüyle yapıldı; ham çıktısı `logs/banking77_shuffled_online_and_neural.txt` dosyasında. O koşunun çevrimiçi satırları, JSON'lu yeniden koşuyla birebir aynı.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `banking77_shuffled_online.json` · `logs/banking77_shuffled_online.txt` | Karışık akış, çevrimiçi öğrenci |
| `banking77_sorted_online.json` · `logs/banking77_sorted_online.txt` | Sıralı akış (stres testi) |
| `logs/banking77_shuffled_online_and_neural.txt` | Karışık akış + yerleşik nöral öğrenci |

## Tekrar üretme

```bash
python examples/benchmark_banking77.py --out banking77_shuffled_online.json
python examples/benchmark_banking77.py --sorted --out banking77_sorted_online.json
python examples/benchmark_banking77.py --neural --epochs 6      # pip install -e '.[neural]' gerekir
```
