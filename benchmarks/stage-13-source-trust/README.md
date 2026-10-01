# Aşama 13 — Kaynağa göre güven (K1)

**Hipotez** ([RESEARCH.md](../../RESEARCH.md), K1): Her etiketleyicinin benzer girdilerde *diğer* etiketleyicilerle ne sıklıkla çeliştiği, medyan etiketleyiciye göre izlenir. Bu bilgi tek kaynaktan gelen hatalı etiket patlamasını herkesin birlikte yaptığı anlam değişikliğinden (drift) ayırabilir.

**Kabul ölçütü:**
- Patlamanın diğer sınıflara hasarı +600 etikette 0,5 puanın altında olmalı.
- Drift ve karışık doğruluk Stage 06 seviyesinde kalmalı.

**Sonuç: kısmen sağlandı.**
- Patlama ölçütü sağlandı.
- Drift'te küçük ama tutarlı bir kötüleşme var.
- `source_trust` **isteğe bağlı** kalıyor, varsayılanı kapalı.

**Kod:** `desic/core/patches.py` (`SourceTrust`). Tam ölçüm commit `7042909`.

**Senaryolar:** Patlama ve drift akışları artık 6 etiketleyiciyle sırayla (round-robin) etiketleniyor.
- Patlamadaki 30 hatalı etiket bu etiketleyicilerden birinden (`a0`) geliyor. Bu etiketleyici öncesinde ve sonrasında dürüst etiketler de veriyor.
- Kaynak güveni kapalı öğrenciler etiketleyici kimliğini yok sayıyor. Bu yüzden önceki aşamaların sonuçları değişmiyor.

## Mekanizma

- Yeni bir etiket, benzerliği ≥ 0,5 olan ve başka bir etiketleyiciden gelen her kayıtla karşılaştırılır. Uyuşma ya da çelişki iki etiketleyiciye de yazılır.
- Sayımlar zamanla söner (yarı ömür 300 etiket).
- **Ağırlık** = medyan çelişki oranı / bu etiketleyicinin çelişki oranı (en fazla 1). Az veride değer medyana doğru çekilir.
- Ağırlık üç yerde kullanılır: patch oylarında, kalıcı modele pekiştirmede ve tekrar çalışmada.
- Pekiştirmede, etiketleyicinin etiket deneme süresindeyken aldığı **en düşük** ağırlık kullanılır. İlk sürümde bu yoktu: saldırgan sonradan dürüst etiketlerle güvenini toparlıyor, hatalı etiketler bu toparlanmanın ardından kalıcı modele tam ağırlıkla geçiyordu.
- Geri alma, etiketin kaynak istatistiklerine katkısını da geri alır. Bir istisna var: geri alınan etiketlerin oluşturduğu "en düşük ağırlık" kayıtları silinir, ama aradaki dönemin karşı-olgusal değerleri yeniden hesaplanmaz. Bu, pekiştirme ağırlığında küçük bir yaklaşıklık bırakıyor.

## Sonuçlar (3 tohum)

| | Stage 06 (kaynak güveni yok) | K1 v1 (anlık ağırlık) | **K1 v2 (en düşük ağırlık)** |
|---|---|---|---|
| Patlama: diğer sınıflara hasar, +600 | 1,01 ± 0,12 puan | 0,66 (0,33 / 1,09 / 0,56) | **0,42 ± 0,27** ✅ |
| Patlama: hasar, hemen sonra | 0,51 | — | 0,48 |
| Drift yarı ömrü | 3.417 ± 290 | — | 3.583 ± 290 |
| Drift: değişen sınıflar, sonda | %61,3 ± 6,0 | — | %59,3 ± 7,2 (3 tohumun 3'ünde 0,7–3,3 puan düşük) |
| Karışık, sıralı, gürültü, öğretmen, bütçe | — | — | birebir aynı (bu akışlarda etiketleyici yok) |

## Tolerans denemesi (3 tohum)

Fikir: ancak medyanın N katından fazla çelişen etiketleyici cezalandırılsın. Böylece drift sırasında rastgele biraz yüksek çıkan dürüst etiketleyiciler etkilenmez.

| Tolerans | Patlama hasarı +600 | Drift sonda |
|---|---|---|
| 1,0 (v2) | 0,42 | %59,3 |
| 1,3 | 1,04 | %61,3 |
| 1,6 | 1,08 | %61,3 |

Tolerans drift'i tamamen düzeltiyor, ama korumayı da tamamen kaldırıyor.

## Neden temiz bir ayrım yok

Patlamadan önce ve hemen sonra etiketleyicilerin uyuşma oranları (tohum 1):

| Benzerlik eşiği | Dürüst etiketleyiciler arası uyuşma | Saldırgan, patlamadan sonra | Saldırganın ağırlığı |
|---|---|---|---|
| 0,5 | %61–65 | %53 | 0,86 |
| 0,7 | %68–82 | %71 | 1,00 |
| 0,85 | %78–92 | %62 | 0,86 |

- **Düşük eşikte** sinyal çok gürültülü. Banking77'de 77 ince niyet var; benzer görünen mesajların ~%40'ı gerçekten farklı etiketli. Bu yüzden dürüst etiketleyiciler bile sık "çelişiyor".
- **Yüksek eşikte** karşılaştırma sayısı çok az. Dürüst etiketleyiciler arasındaki rastgele fark (%78–92), saldırganın sapması kadar büyük.
- Sonuçta saldırganın sapması (~1,2× medyan), drift sırasında dürüst etiketleyicilerin rastgele sapmasıyla aynı bantta kalıyor. Bu veri setinde, etiketler arası uyuşmaya dayalı bir kaynak güveninin ayırma gücü bununla sınırlı.

## Değerlendirme

- **K1 ilk kez iki şeyi birlikte iyileştirdi:**
  - Patlamanın pekiştirme sonrası hasarı %58 azaldı.
  - Karışık, sıralı ve gürültü senaryolarına hiç dokunmadı.

  Stage 09 ve 10'daki denemeler bunun tersini yapıyordu: hasarı azaltan her şey drift'i ciddi bozuyordu (%65,8 → %50).
- **Bedeli:** drift'te sonda ~2 puan. Bu, tohumlar arası standart sapmanın (~6 puan) içinde ama yönü tutarlı.
- **Daha güçlü bir kaynak sinyali için** etiketler arası uyuşmadan başka bir kanıt gerekiyor:
  - Etiketleyicinin geri alınan etiketlerinin oranı (insanlar kimin etiketini düzeltiyor).
  - İnsan tarafından doğrulanmış örneklerde etiketleyicinin doğruluğu (altın standart kontrolleri).

  Bu iki sinyal servis tarafında doğal olarak oluşuyor. Bir sonraki adım, `annotator` kimliğini API'ye ve geri bildirim kaydına eklemek.

## Dosyalar

- `results_sources.json`: K1 v2, tüm senaryolar
- `drift_patch_rerun.json`: karşılaştırma
- `logs/sweep_v1_seed.txt`, `logs/sweep_v2_lowest.txt`, `logs/sweep_v3_tolerance.txt`
