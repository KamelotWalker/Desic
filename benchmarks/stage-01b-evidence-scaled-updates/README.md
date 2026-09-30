# Aşama 01b — Kanıtla ölçeklenen güncellemeler (kabul edildi)

**Amaç:** İlk kullanıcı denemesinde bulunan aşırı güven hatasını ([`stage-01a`](../stage-01a-adagrad-g0-rejected/) — kök neden ve reddedilen deneme) Banking77'deki başarımı bozmadan düzeltmek.

- **Kod:** commit `babfa25`
- **Ortam:** Python 3.11, 4 çekirdek Intel Xeon @ 2.80 GHz, GPU yok

## Değişiklikler

1. **Adım, etiketin ağırlığı ve bilgi değeriyle ölçekleniyor.** Bilgi değeri: tek bir cevaba ne kadar yoğunlaştığı, eşit dağılıma göre. Kesin (tek cevaplı) insan ve veri seti etiketleri eskisiyle tamamen aynı davranıyor. %55'lik bir öğretmen etiketi neredeyse hiç adım attırmıyor, %90'lık bir öğretmen etiketi ise hâlâ öğretiyor.
2. **"Tanıdıklık" yalnızca gerçekten öğrenilmiş kelimeleri sayıyor** (herhangi bir cevapta |ağırlık| ≥ 0,05). Yazı-tura bir etiketin dokunduğu kelimeler tanıdık sayılmıyor.
3. **Tanıdık olmayan girdi her zaman çekimser kalıyor** (tanıdıklık < %50 ve önceden eğitilmiş encoder yok). Denemede %38,5 tanıdıklıkta %64,1 güvenle yanlış cevap verilmişti; artık soru öğretmene gidiyor.
4. **Tek metin kolonu her zaman düz metin olarak eğitiliyor.** Kısa mesajlar önceden `{"state": ...}` nesnesine dönüşüyordu, karar anında ise düz metin geliyordu.

## Kullanıcı denemesindeki senaryo (birim testleriyle doğrulandı)

`urgent` sorusu, demo destek talepleriyle eğitilmiş; mesaj "Merhaba, erken rezervasyon yaptırmak istiyorum":

| | Etiketten önce | Stage 00 | Stage 01b |
|---|---|---|---|
| Öğretmen %55 "true" dedikten sonra, aynı mesaj | 0,37 | **0,94** | 0,35 |
| Aynı etiketten sonra ilgisiz "Rezervasyonumu iptal etmek istiyorum" | 0,68 | 0,81 | 0,68 |
| Öğretmen %90 "true" dedikten sonra | 0,37 | — | 0,70 |
| İnsan "true" düzeltmesinden sonra | 0,37 | 0,97 | 0,97 |

## Banking77 (regresyon kontrolü)

Banking77'de yalnızca kesin etiketler var, bu yüzden sonuçların Stage 00 ile birebir aynı olması bekleniyordu ve öyle çıktı:

| Eğitim | Stage 00 | Stage 01b |
|---|---|---|
| 10 örnek/sınıf | %57,2 · ECE 0,122 | %57,2 · ECE 0,122 |
| 25 örnek/sınıf | %72,8 · ECE 0,019 | %72,8 · ECE 0,019 |
| 50 örnek/sınıf | %82,6 · ECE 0,011 | %82,6 · ECE 0,011 |
| Tüm veri, karışık | %88,5 · ECE 0,013 | %88,5 · ECE 0,013 |
| Tüm veri, sıralı akış | %15,8 · ECE 0,474 | %15,8 · ECE 0,474 |

## Açık kalanlar

- **Sıralı akıştaki çöküş** (%15,8) bu aşamada ele alınmadı; sonraki aşamanın hedefi.
- Öğretmen etiketlerinin etkisi Banking77'de ölçülmüyor. Öğretmenli bir benchmark (etiketsiz veri + öğretmen + seyrek insan düzeltmesi) gerekiyor.
- **Mevcut sorular eski davranışla devam eder.** Codespaces'teki soruların yeni kuralla öğrenmesi için soru sayfasında *Feedback log → Rebuild student from log* ile kayıttan yeniden kurulmaları gerekiyor.

## Tekrar üretme

```bash
python examples/benchmark_banking77.py --out banking77_shuffled_online.json
python examples/benchmark_banking77.py --sorted --out banking77_sorted_online.json
pytest tests/test_core.py -k "teacher_label or human_correction or unfamiliar"
```
