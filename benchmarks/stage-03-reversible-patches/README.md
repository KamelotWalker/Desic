# Aşama 03 — Geri alınabilir patch katmanı (Faz 2)

**Amaç:** Stage 02'nin bulduğu iki zayıflık: (1) hatalı etiketi geri almanın tek yolu tüm kaydı yeniden oynatmak, hasar da diğer sınıflara yayılıyor; (2) sıralı akışta eski sınıflar unutuluyor. Bunları karışık akışta başarımı bozmadan çözmek.

- **Kod:** `desic/core/patches.py` (`PatchedTask`), harness'te öğrenci adı `desic+patch`. Son sonuçlar commit `30bf8a9`. Ablasyon (ortak kapı) ve temel model, commit `aebf28b` ile ölçüldü; temel modelin kodu bu iki commit arasında değişmedi.
- **Ortam:** Python 3.11, 4 çekirdek Intel Xeon @ 2.80 GHz, GPU yok. 7 senaryo × 3 tohum.
- **Karşılaştırma:** iki öğrenci birebir aynı akışları gördü. Temel model bu aşamada yeniden ölçüldü, çünkü patlama senaryosu artık diğer sınıflara hasarı ayrıca raporluyor; diğer sayılar Stage 02 ile aynı.

## Nasıl çalışıyor

Temel model (Stage 02'deki öğrenci) değişmedi; önüne bir katman kondu:

1. **Patch:** her etiket anında bir kayıt olur: durum, cevap ve kanıt (kaynak, ağırlık, ref, zaman). Yalnızca *benzer* girdileri etkiler (kosinüs benzerliği, en yakın 10 kayıt).
2. **Deneme süresi (probation):** temel model bir etiketi ancak 500 olay sonra öğrenir. O zamana kadar geri almak birebir ve O(1): temel model o etiketi hiç görmemiştir.
3. **Güven:** bir patch'in yakınına düşen sonraki etiketler onu destekler (aynı cevap) ya da çürütür (başka cevap). Çürütülen patch'in oyu ve tekrar edilme olasılığı düşer.
4. **Kapı:** patch'e mi temel modele mi ne kadar güvenileceği öğrenilir. Her bağlam hücresi için ayrı bir Hedge karışımı var: hücre en yakın patch'in benzerliği, patch'lerle temel modelin hemfikir olup olmaması ve patch'lerin önerdiği cevaptan oluşur. Yeni bir cevap ortak hücreden başlar. Üstte ayrıca bir sıcaklık kalibrasyonu var.
5. **Pekiştirme (consolidation):** temel model bir etiketi öğrenirken eski, güvenilir kayıtlardan sınıf dengeli bir örneği de yeniden çalışır (rehearsal).
6. **Kontrol noktası:** temel model her 1.000 pekiştirmede kaydedilir. Pekiştirilmiş bir etiketi geri almak, en yakın önceki kontrol noktasından itibaren yeniden oynatır.
7. Bir etiketin temel model dışında değiştirdiği her şey (komşuların güveni, kapı, kalibrasyon) kayıt bazında loglanır ve geri alınınca o da geri alınır.

**Doğrulama (unit test):** deneme süresindeki etiketler geri alınınca cevaplar etiketten öncekiyle birebir aynı (10⁻⁹ toleransla). Pekiştirilmiş etiketler geri alınınca temel model, o etiketleri hiç görmemiş bir modelle birebir aynı ve yeniden oynatma sıfırdan değil, kontrol noktasından başlıyor.

## Sonuçlar (3 tohum, ortalama ± std)

| Senaryo | Metrik | Temel model | Patch katmanı | |
|---|---|---|---|---|
| **Karışık** | Doğruluk | %88,75 ± 0,23 | %88,86 ± 0,13 | ≈ (regresyon yok) |
| | Log loss / ECE | 0,423 / 0,015 | 0,410 / 0,013 | ≈ |
| | Risk@%80 / AURC | %3,26 / 0,019 | %3,03 / 0,018 | ≈ |
| | Kümülatif log loss | 0,878 | 0,979 | **kötü** (aşağıya bkz.) |
| **Sıralı** | Doğruluk | %13,7 ± 1,5 | **%80,2 ± 0,4** | çok iyi |
| | Forgetting index | 0,63 ± 0,04 | **0,097 ± 0,007** | çok iyi |
| | ECE | 0,50 | 0,24 | iyi ama hâlâ kötü kalibre |
| **Patlama** (30 hatalı etiket) | Kurban, hemen sonra | %0,8 | **%65 ± 15** | iyi |
| | Kurbanın saldırı etiketini alan mesajları | %93 | %11 | iyi |
| | Diğer sınıflardaki hasar | 3,8 ± 0,9 puan | **0,5 ± 0,4 puan** | çok iyi |
| | Geri alma süresi | 29,6 sn | **0,105 sn** | ~280× hızlı |
| | Geri almada yeniden oynatılan olay | 5.001 | **0** | |
| | Geri alma sonrası test doğruluğundaki fark | 0 | −0,12 ± 0,27 puan | ≈ (aşağıya bkz.) |
| | Akış sonunda kurban | %75 | %76,7 | ≈ |
| **%1 / %5 gürültü** | Doğruluk | %88,5 / %87,9 | %88,7 / %88,0 | ≈ |
| | Yalanı tekrarlama | %6,4 / %6,9 | %22,8 / %23,4 | **kötü** |
| **Drift** | Adaptation half-life | 3.000 ± 250 | 3.417 ± 290 | **kötü** |
| | Değişen sınıflar, sonda | %64,8 | %59,9 | **kötü** |
| | Diğer sınıflar: en düşük / son | %77,3 / %88,3 | %79,8 / %89,5 | iyi |
| **Öğretmen** | Teacher dependency (son 500) | %46,3 | %41,1 | iyi |
| | Kendi cevaplarının doğruluğu (son 500) | %92,4 | %88,9 | **kötü** |
| | Test doğruluğu / ECE | %73,6 / 0,115 | %71,8 / 0,046 | doğruluk −1,9 puan, kalibrasyon iyi |
| **Maliyet** | Toplam süre (21 koşu, 4 çekirdek) | 9,1 dk | 17,0 dk | ~2× yavaş |

## Bulgular

**Başarılı olanlar**

- **Geri alma artık ucuz ve birebir.** Patlama deneme süresi içindeyken geri alınınca hiçbir olay yeniden oynatılmıyor ve süre olay sayısından bağımsız: 0,1 sn. Temel modelde süre kayıtla doğrusal büyüyor: burada 5.000 olay 30 sn, 1 milyon olayda saatler sürer.
- **Patlamanın diğer sınıflara hasarı neredeyse yok** (3,8 → 0,5 puan). Bunu sağlayan, kapının cevap bazında ayrılması. Ablasyonda ortak kapı ile hasar 3,9 ± 3,1 puandı, bir tohumda 7,5 puan: saldırı sırasında kapı "uyuşmazlıkta patch'e güven" diye öğreniyor ve bunu bütün sınıflara uyguluyordu.
- **Sıralı akışta unutma büyük ölçüde çözüldü** (%13,7 → %80,2). Teşhis: temel modelin bellek uzmanı yalnızca son 2.000 örneği tutuyordu (FIFO). Lineer model de her yeni sınıf bloğunda eskilerin ağırlıklarını eziyordu (tek başına %14). Patch belleği eski örnekleri atmıyor; rehearsal ise temel modeli eski sınıflarda tutuyor. Rehearsal kapatılınca (3k alt örneklemde) sonuç değişmedi, yani kazancın asıl kaynağı bellek. Not: bu testlerde bellek kapasitesi (20.000) akıştan büyük olduğu için sınıf dengeli atma hiç devreye girmedi; o ayrıca, daha uzun bir akışla ölçülmeli.
- **Karışık akışta regresyon yok:** doğruluk, log loss, ECE ve seçici risk aynı ya da biraz iyi.

**Kötüleşenler ve sebepleri**

- **Drift daha yavaş** (half-life 3.000 → 3.417). Teşhis (tohum 0): kapı drift sırasında çoğunlukla temel modele güveniyor. Patch'ler ise eski anlamdaki eski kayıtlarla yeni kayıtlar arasında bölünüyor (akış sonunda patch'ler tek başına %32). Temel model de yeni etiketleri 500 olay geç öğreniyor ve eski kayıtları da tekrar ediyor. Güven mekanizması bunu yeterince hızlı telafi etmiyor. Diğer sınıflar ise daha kararlı.
- **Aynı mesajda yanlışı tekrarlama** %7'den %23'e çıktı. Bu, "bir düzeltme o mesaj için anında tutar" davranışının diğer yüzü: patch'ler yerel etikete güveniyor. Yanlış etiketin çaresi artık ucuz olan geri alma; ama otomatik tespit yok.
- **Kümülatif log loss kötü** (0,88 → 0,98). Akışın başında patch'ler temel modelden daha az iyi tahmin ediyor ve temel model 500 olay geriden geliyor. Final kalite aynı ama akış boyunca ödenen bedel daha yüksek.
- **Öğretmenli soğuk başlangıçta** model daha çok kendisi cevaplıyor (bağımlılık %46 → %41) ama daha az doğru (%92 → %89). Daha az öğretmen etiketi aldığı için test doğruluğu 1,9 puan düşük. Kalibrasyon ise belirgin şekilde iyi (ECE 0,115 → 0,046). Bu bir risk dengesi; Faz 1'deki risk bütçesi tam bunu ayarlamak için.
- **~2× yavaş:** her karar ve etiket için komşu araması ve pekiştirmede rehearsal.

**Notlar**

- *Geri alma sonrası −0,12 puan:* patlama sürerken (30 olay) deneme süresini dolduran 30 temiz etiket temel modele geçti; geri almada bunlar doğal olarak kalıyor. Fark bundan ve rehearsal'ın rastgeleliğinden geliyor. Birebirlik unit testlerde karşı-olgusal (o etiketleri hiç görmemiş) bir modelle ayrıca doğrulandı.
- *Patlamada toparlanma yarı ömrü* (2.100 ± 2.300) bu aşamada karşılaştırılabilir değil: patch katmanında düşüş çok küçük (örneğin %82,5 → %80), "yarı yol" hedefi de birkaç test mesajına denk geliyor.

## Ablasyon (3k alt örneklem, tohum 0 — eğilim içindir, kesin değildir)

| Varyant | Karışık doğruluk / ECE | Sıralı / FI | Patlama: diğer sınıflara hasar |
|---|---|---|---|
| Temel model | %77,4 / 0,018 | %36 / 0,53 | 14,7 puan |
| Patch, ilk sürüm (güven yok, kalibrasyon yok) | %78,9 / 0,055 | %68,5 / 0,10 | 1,3 puan |
| + zamanla oy azaltma (yarı ömür 1.000) | %78,8 / 0,050 | %62,0 / 0,21 | 3,6 puan — **elendi** |
| rehearsal yok | %76,9 / 0,057 | %68,0 / 0,10 | 0,9 puan |
| deneme süresi 200 | %79,4 / 0,070 | %69,0 / 0,10 | 4,3 puan |
| + güven + son kalibrasyon | %79,2 / 0,043 | %68,9 / 0,11 | 1,0 puan |

Tam veride (3 tohum) ortak kapı ile cevap bazlı kapı: `results_patch_shared_gate.json` ve `results_patch.json`.

## Sonraki adımlar

| Hedef | Neden |
|---|---|
| Drift: yerel drift tespiti, çürütülen patch'lerin hızlı pekiştirilmesi | Half-life 3.417; hedef temel modelden belirgin şekilde kısa |
| Yerel uyuşmazlıklarda yanlış etiket şüphesi ve insana sorma | Yalan tekrarlama %23; geri alma ucuz ama tespit yok |
| Risk bütçesi (Faz 1) | Öğretmen senaryosunda kapsam–doğruluk dengesi elle ayarlanmamalı |
| Servise ve dashboard'a entegrasyon | Patch katmanı şu an yalnızca çekirdekte ve harness'te; API ve dashboard hâlâ temel modeli kullanıyor |

## Dosyalar

- `results_desic.json`: temel model (commit `aebf28b`)
- `results_patch_shared_gate.json`: patch katmanı, ortak kapı (ablasyon, commit `aebf28b`)
- `results_patch.json`: patch katmanı, cevap bazlı kapı (commit `30bf8a9`)
- `logs/*.txt`: konsol çıktıları

## Tekrar üretme

```bash
desic eval --learner desic --seeds 3 --out results_desic.json
desic eval --learner desic+patch --seeds 3 --out results_patch.json
pytest tests/test_patches.py
```
