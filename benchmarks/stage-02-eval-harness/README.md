# Aşama 02 — Değerlendirme düzeneği (Faz 0)

**Amaç:** Önce ölç, sonra değiştir. Bundan sonraki her değişiklik (geri alınabilir patch katmanı, bağlamsal router, …) aynı akışlarla bu tabloya karşı ölçülecek. Bu aşamada modelde hiçbir değişiklik yok, sadece ölçüm var.

- **Kod:** commit `ebeb5a7` (`desic/eval/`, `desic eval` komutu). JSON'daki `meta.commit` `5deae56` gösteriyor: commit, koşunun sonunda okunuyordu ve arada yalnızca dokümantasyon commit'lendi. Kod iki commit'te birebir aynı; commit artık koşunun başında okunuyor.
- **Ortam:** Python 3.11, 4 çekirdek Intel Xeon @ 2.80 GHz, GPU yok. 7 senaryo × 3 tohum, 4 paralel işçi: 9,4 dakika.
- **Veri:** Banking77, 10.003 eğitim / 3.080 test, 77 niyet.
- **Senaryo ve metrik tanımları:** [`../README.md`](../README.md#akış-metrikleri-aşama-02den-itibaren-desic-eval).

Tüm değerler 3 tohumun ortalaması ± standart sapmasıdır.

## Özet

| Senaryo | Sonuç | Değerlendirme |
|---|---|---|
| Karışık akış | %88,8 ± 0,2 doğruluk, ECE 0,015, Risk@%80 %3,3 | Güçlü |
| %1 / %5 hatalı etiket | Doğruluk %88,5 / %87,9; ECE 0,022 / 0,085 | Doğruluk dayanıklı; %5'te kalibrasyon bozuluyor |
| Sıralı akış | %13,7 doğruluk, forgetting index 0,63 | **Çöküş** |
| 30 hatalı etiketlik patlama | Kurban sınıf %77 → %1; ~2.800 etiket sonra ancak yarısı geri geliyor; geri alma = 5.000 olayın yeniden oynatılması | **Zayıf** |
| Drift (10 sınıfın anlamı değişti) | Yarı toparlanma ~3.000 etiket; akış sonunda %65 (önce %84) | **Yavaş**; drift dedektörü hiç tetiklenmedi |
| Öğretmenli soğuk başlangıç | Öğretmene giden oran %95 → %46; kendi cevapladıklarında %92 doğru | Öğreniyor, ama kalibrasyon zayıf (ECE 0,115) |

## 1. Karışık akış (`shuffled`)

| Etiket sayısı | Doğruluk | ECE | Kapsam | Cevaplanan doğruluk |
|---|---|---|---|---|
| 500 | %45,2 | 0,133 | %29,6 | %62,5 |
| 1.000 | %59,2 | 0,037 | %44,5 | %84,6 |
| 2.500 | %75,1 | 0,013 | %72,3 | %88,3 |
| 5.000 | %80,1 | 0,019 | %80,4 | %89,6 |
| 10.003 | %88,5 | 0,013 | %89,7 | %93,7 |

(Tablo tohum 0'a ait. Tohum 0 ile Stage 00'daki %88,5 birebir tekrar üretildi.)

3 tohum ortalaması:

| Metrik | Değer |
|---|---|
| Doğruluk | %88,75 ± 0,23 |
| Log loss | 0,423 ± 0,005 |
| ECE | 0,015 ± 0,007 |
| Risk@%80 | %3,26 ± 0,02 |
| AURC | 0,019 |
| Varsayılan çekimserlikle kapsam | %88,2 ± 1,7 |
| O cevapların doğruluğu | %94,3 ± 0,6 |
| Kümülatif log loss (akış boyunca) | 0,878 ± 0,008 |

## 2. Hatalı etiket (`noise-1%`, `noise-5%`)

| | Temiz | %1 yanlış (~98 etiket) | %5 yanlış (~515 etiket) |
|---|---|---|---|
| Doğruluk | %88,75 | %88,54 ± 0,21 | %87,93 ± 0,31 |
| Log loss | 0,423 | 0,430 | 0,504 |
| ECE | 0,015 | 0,022 ± 0,011 | **0,085 ± 0,019** |
| Risk@%80 | %3,26 | %3,29 | %3,88 |
| Yalan tekrarlama | — | %6,4 ± 4,0 | %6,9 ± 1,6 |

- **Rastgele hatalara karşı doğruluk dayanıklı.** Yanlış etiketlendiği mesajlar yeniden sorulduğunda model yalnızca %4–11'inde yanlışı tekrarlıyor, %78–88'inde doğru cevabı veriyor. Diğer uzmanlar bellek uzmanının ezberini bastırıyor.
- **Kalibrasyon hassas.** %5 gürültüde ECE yaklaşık 6 katına çıkıyor: model doğruluğundan daha az emin konuşmaya başlıyor. Sebebi büyük ihtimalle sıcaklık kalibratörünün yanlış etiketlere de uydurulması.

## 3. Sıralı akış (`sorted`)

| Metrik | Değer |
|---|---|
| Final doğruluk | %13,7 ± 1,5 |
| ECE | 0,50 ± 0,03 |
| Forgetting index | **0,63 ± 0,04** |
| Kümülatif log loss | 0,362 ± 0,005 |

- Her sınıf grubu öğrenildiği anda iyi: en yeni grubun doğruluğu %41–90. Ama önceki sınıflar ortalama 63 puan kayboluyor.
- **Metodolojik uyarı:** kümülatif log loss burada karışık akıştakinden bile *düşük* (0,36'ya karşı 0,88). Sıralı akışta bir sonraki etiket neredeyse her zaman bir öncekiyle aynı; "son gördüğünü tekrarla" stratejisi prequential metriği kandırıyor. Bu yüzden sıralı akışta prequential kayıp tek başına kullanılmamalı; test kümesiyle birlikte okunmalı.

## 4. Hatalı etiket patlaması (`burst`)

5.001 temiz etiketten sonra, bir A sınıfının 30 mesajı art arda B olarak etiketleniyor. Bu, akışın %0,3'ü. Ardından akışın temiz kalanı geliyor.

| Tohum | Kurban A → saldırı etiketi B | Kurban önce → sonra | Test genel düşüş | Bunun kurban dışı kısmı |
|---|---|---|---|---|
| 0 | top_up_reverted → virtual_card_not_working | %82,5 → %0 | 5,8 puan | 4,7 puan |
| 1 | pin_blocked → exchange_charge | %77,5 → %2,5 | 4,3 puan | 3,3 puan |
| 2 | verify_my_identity → verify_source_of_funds | %70 → %0 | 4,1 puan | 3,2 puan |

| Metrik | Değer |
|---|---|
| Kurbanın test mesajlarından saldırı etiketini alan | %93 ± 7 |
| Toparlanma yarı ömrü (geri almadan, temiz etiketlerle) | 2.833 ± 680 etiket |
| Akış sonunda kurban doğruluğu | %75 ± 10 |
| Bugünkü geri alma maliyeti | 5.001 olayın yeniden oynatılması, 29,4 ± 0,5 sn |

- **30 etiket bir sınıfı tamamen ele geçiriyor.** Temiz etiketlerle toparlanma, akışın geri kalanının yarısından fazlasını alıyor (~2.800 etiket; bunların içinde kurban sınıfından yalnızca 22–31 örnek var).
- **Hasar kurbanla sınırlı değil.** Genel düşüşün yaklaşık dörtte üçü, yani 3–5 puan, başka sınıflarda. Kurbanın 40 test mesajının ağırlığı ancak ~1 puan ediyor. Saldırı etiketi, ortak kelimeler ve bias üzerinden alakasız mesajlara da yayılıyor. Patch katmanı için önemli bir bulgu: düzeltmenin *kapsamı* sınırlanmalı.
- **Geri almak bugün mümkün ama pahalı.** Tüm kaydın yeniden oynatılması gerekiyor. Maliyet kayıtla doğrusal büyüyor: burada 5.000 olay 29 sn, 1 milyon olayda saatler sürer. Geri alındığında sonuç, patlama hiç olmamış gibi birebir aynı.

## 5. Drift (`drift`)

5.001 etiketten sonra 10 sınıfın anlamı döndürülüyor: c₁ → c₂ → … → c₁₀ → c₁. Geri kalan 67 sınıf aynı kalıyor.

| Metrik | Değer |
|---|---|
| Değişen sınıflar, drift öncesi (eski anlam) | %83,7 ± 4,0 |
| Değişen sınıflar, drift anında (yeni anlam) | %0 |
| Adaptation half-life | **3.000 ± 250 etiket** (içinde değişen sınıflardan ~400) |
| Değişen sınıflar, akış sonunda | %64,8 ± 4,1 |
| Diğer sınıflar: önce / en düşük / son | %83,3 / %77,3 / %88,3 |

- **Yeni anlamı öğrenmek yavaş.** Model eski anlamı hafızasından silemiyor. Eski kanıt yeni kanıtla ancak adım adım eziliyor. Muhtemel sebepler (henüz ayrıca ölçülmedi): AdaGrad'ın öğrenme hızı zamanla küçülüyor ve bellek uzmanı eski örnekleri tutuyor.
- **DDM drift dedektörü hiç tetiklenmedi (0/3).** Değişen sınıflar etiketlerin yalnızca %13'ü. Genel hata oranı bu yüzden dedektörün eşiğini aşacak kadar yükselmiyor. Global bir dedektör yerel drift'i kaçırıyor.
- Diğer sınıflar drift sırasında 6 puana kadar düşüyor, sonra daha fazla veriyle toparlanıyor.

## 6. Öğretmenli soğuk başlangıç (`teacher`)

5.000 etiketsiz mesaj. Model çekimser kaldığında öğretmen etiketliyor. Öğretmen simüle ve kalibre: güveni 0,6 / 0,75 / 0,9 / 0,97, o olasılıkla doğru; gerçekleşen doğruluk %80. İnsan ise mesajların %5'ini kontrol ediyor.

| Karar sayısı | 500 | 1.000 | 2.000 | 3.000 | 4.000 | 5.000 |
|---|---|---|---|---|---|---|
| Öğretmene giden oran (tohum 0) | %92 | %95 | %93 | %77 | %60 | %49 |
| Kendi cevapladığının doğruluğu | %51 | %54 | %89 | %94 | %94 | %95 |

| Metrik | Değer |
|---|---|
| Teacher dependency, ilk / son 500 karar | %94,8 → **%46,3 ± 4,9** |
| Son pencerede kendi cevaplarının doğruluğu | %92,4 ± 3,6 |
| Toplam öğretmen çağrısı / insan etiketi | 3.594 / 256 |
| Test: doğruluk / ECE | %73,6 / **0,115** |
| Test: kapsam / cevaplanan doğruluk | %54,3 / %92,3 |

- Model öğretmenden öğreniyor ve bağımlılık yarıya iniyor. Kendi verdiği cevaplar, %80 doğru bir öğretmenden daha doğru (%92), çünkü sadece emin olduğunda cevap veriyor.
- Aynı sayıda temiz etiketle (5.000) test doğruluğu %80,1 iken burada %73,6. Öğretmenin %20 hatası bir tavan koyuyor.
- **Kalibrasyon zayıf (ECE 0,115).** Sıcaklık kalibratörü sadece insan etiketlerinden (~250) öğreniyor; muhtemel sebep, yumuşak öğretmen etiketlerinin modeli fazla temkinli yapması (henüz ayrıca ölçülmedi).

## Bundan sonrası için hedefler

Bu ölçümler Faz 2'nin (geri alınabilir patch katmanı) başarı ölçütü:

| Hedef | Stage 02 | Ölçüt |
|---|---|---|
| Patlamayı geri alma maliyeti | 5.001 olay, 29 sn | Patch sayısıyla orantılı (≪ 1 sn), sonuç yeniden oynatmayla aynı |
| Patlamanın kurban dışı hasarı | 3–5 puan | ~0 (kapsamlı düzeltme) |
| Drift yarı ömrü | 3.000 etiket | Belirgin şekilde daha kısa, diğer sınıflar kararlı |
| Forgetting index (sıralı) | 0,63 | Düşmeli |
| Karışık akış | %88,75, ECE 0,015 | Bozulmamalı (regresyon yok) |

## Dosyalar

- `results.json`: her senaryo ve tohum için tüm eğriler, ham değerler, `summary` (ortalama ± std).
- `logs/eval.txt`: konsol çıktısı.

## Tekrar üretme

```bash
pip install -e .
desic eval --seeds 3 --out results.json
```
