# Aşama 10 — Hasar yarıçapı pekiştirmeden sonra (B1)

**Soru** ([RESEARCH.md](../../RESEARCH.md), B1): 30 hatalı etiket deneme süresini geçip kalıcı modele pekiştirildiğinde, diğer sınıflara verilen hasar yeniden büyüyor mu?

- **Kod:** burst senaryosu, commit `f4e941a`. Damping denemesi commit `6fd52e5` ile yapıldı (varsayılanı kapalı).
- **Ölçüm yöntemi:** Geri alınmış kopya, yani patlamayı hiç görmemiş bir **ikiz**, saldırıya uğrayan öğrenciyle aynı temiz etiketleri öğrenmeye devam ediyor. Hasar = ikizin diğer sınıflardaki test doğruluğu − saldırıya uğrayanın. Patlamadan hemen sonra, 600 ve 1.000 etiket sonra ölçülüyor. Bu noktada deneme süresi 500 etiket, yani 600'de patlama kalıcı modeldedir.
- **Ortam:** 3 tohum.

## Sonuç: hipotez kısmen doğrulandı

Diğer sınıflardaki hasar, ikize göre (puan, 3 tohum ortalaması):

| | Hemen sonra | +600 etiket | +1.000 etiket |
|---|---|---|---|
| Temel model | **3,8** ± 0,9 | 0,1 ± 0,2 | 0,0 ± 0,1 |
| Patch'li öğrenci | 0,5 ± 0,4 | **1,0** ± 0,1 | 0,1 ± 0,2 |

- **Temel modelde** hasar büyük ama kısa ömürlü: 600 temiz etiket içinde iyileşiyor.
- **Patch'li öğrencide** hasar başta küçük, ama patlama kalıcı modele geçtiğinde ikiye katlanıyor (0,5 → 1,0), sonra iyileşiyor.
- Yani patch katmanı hasarı **yok etmiyor, küçültüp erteliyor.** Tepe hasar temel modelin ~¼'ü, ama 600. etiket civarında temel modelden kötü.
- B1'in "temel model seviyesine (3,8) döner" kısmı doğrulanmadı.

## Denenen düzeltme: pekiştirmede sönümleme (reddedildi)

Fikir: kalıcı model bir kaydı makul bulmuyorsa (o cevaba verdiği olasılık eşiğin altındaysa), kaydı daha düşük ağırlıkla öğrensin.

| | Stage 06 varsayılanı | Eşik 0,2 | Eşik 0,5 |
|---|---|---|---|
| Hasar +600 (3 tohum) | 1,0 puan | **~0** (−0,4 … +0,1) | **~0** (−0,3 … 0) |
| Drift: değişen sınıflar, sonda (tohum 0) | %65,8 | **%50,0** | **%50,0** |
| Drift yarı ömrü (tohum 0) | 3.750 | 4.000 | 4.000 |
| Karışık doğruluk (tohum 0) | %89,1 | **%86,4** | **%85,3** |
| Sıralı doğruluk (tohum 0) | %79,8 | %80,1 | %80,2 |

- Sönümleme pekiştirme sonrası hasarı tamamen yok ediyor.
- Ama drift'e uyumu belirgin şekilde bozuyor (%65,8 → %50) ve karışık akışta 3–4 puan kaybettiriyor. Kalıcı modelin "beklemediği" doğru etiketler de sönümleniyor; zor örnekler ve yeni anlamlar tam da bunlar.
- RESEARCH.md'deki kural ("ikisini birden iyileştirmeyen sürüm kabul edilmez") gereği **reddedildi**.

## Ne öğrendik

- "Kalıcı model bu etiketi beklemiyordu" sinyali hatalı etiketi doğru ama beklenmedik etiketten ayıramıyor. Drift ile saldırı arasındaki gerilim (D1) burada da aynen çıkıyor.
- Ayırt etmek için etiketin kendisinden başka bir bilgi gerekiyor:
  - **Kaynak:** kimin etiketlediği, o kaynağın geçmiş güvenilirliği.
  - **Zamansal bağlam:** aynı kaynaktan kısa sürede gelen çelişkili etiketler.
  - **İnsan onayı:** pekiştirmeden önce şüpheli bölgeler için onay.
- Bu bir sonraki hipotez adayı (kaynak bazlı güven). Hasar zaten küçük (en çok 1 puan) ve geçici; geri alma da her an mümkün. Bu yüzden şimdilik varsayılan değişmiyor.

## Dosyalar

- `burst_desic.json`, `burst_patch.json`: 3 tohum
- `logs/damping_sweep.txt`, `damping_variants.json`: sönümleme denemesi
