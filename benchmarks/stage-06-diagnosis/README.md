# Aşama 06 — Teşhis: kalan zayıflıklar nereden geliyor?

**Amaç:** Stage 05'te tahmine dayalı düzeltmeler işe yaramadı. Bu aşamada tahmin etmiyoruz, ölçüyoruz. Patch'li öğrencinin parçalarını (kalıcı model, patch'ler, aralarındaki kapı) her zayıflıkta ayrı ayrı puanlıyoruz.

- **Kod:** `diagnose.py`. Stage 04 ile aynı ayarlar; drift, %5 gürültü ve öğretmen senaryoları, tohum 0, aynı akışlar.
- **Ham sonuçlar:** `diagnosis.json`.
- **Okuma kılavuzu:** "kalıcı model tek başına" = patch katmanının içindeki temel modelin cevabı. "Düz temel model" = patch katmanı olmadan, tek başına çalışan temel model.

## 1. Yanlış etiketi tekrarlama (%5 gürültü)

Yanlış etiketlenmiş 533 mesaj akış bitince yeniden soruldu:

| | Yanlışı tekrarlıyor |
|---|---|
| Düz temel model | %5,6 |
| Patch'li öğrenci (birleşik cevap) | %20,6 |
| ↳ içindeki kalıcı model tek başına | **%20,1** |
| ↳ patch'ler tek başına | %34,7 |
| ↳ kapının patch'lere verdiği ağırlık | ortalama **0,12** |

**Bulgu:** Yanlışı tekrarlayan asıl parça patch'ler değil, **içerideki kalıcı model**: düz temel modelin dört katı (%20'ye karşı %5,6). Kapı patch'lere zaten az güveniyor (0,12). Kalıcı model ile düz temel model arasındaki tek fark, pekiştirme sırasındaki **tekrar çalışma** (rehearsal): eski kayıtlar kalıcı modele yeniden gösteriliyor. Yanlış etiketli bir mesaj da tekrar gösterildikçe kalıcı model onu ezberliyor.

## 2. Drift (10 sınıfın anlamı değişiyor)

Değişen sınıfların test mesajlarında, yeni anlama göre doğruluk:

| Drift'ten sonraki etiket sayısı | 2.000 | 3.000 | 4.000 | 5.002 |
|---|---|---|---|---|
| Düz temel model | %21,8 | %42,0 | %68,8 | %67,3 |
| Patch'li, birleşik | %9,3 | %31,5 | %49,5 | %65,8 |
| ↳ kalıcı model tek başına | %9,3 | %34,0 | %49,8 | %64,8 |
| ↳ patch'ler tek başına | %16,6 | %27,7 | %42,2 | %51,9 |
| ↳ ikisinden biri doğru (üst sınır) | %23,5 | %44,3 | %62,0 | %73,8 |
| Patch oyunda eski anlamın payı | %45 | %39 | %32 | %28 |
| Tekrar çalışma kapalı, birleşik | %10,8 | %33,0 | %50,3 | %67,3 |
| Deneme süresi 20, birleşik | %19,8 | %39,3 | %57,0 | %64,3 |

**Bulgular:**

- **Yavaşlığın ana kaynağı kalıcı model:** birleşik cevap hemen hemen onunla aynı. Patch'ler tek başına daha da kötü, çünkü sonda bile oylarının %28'i eski anlamdan geliyor.
- **Tekrar çalışma, kalıcı modelin sonunda ulaştığı yeri düşürüyor:** kalıcı model tek başına, tekrar çalışma kapalıyken %68,5 (düz temel modelden bile biraz iyi), açıkken %64,8. Eski anlamdaki kayıtlar tekrar gösterildikçe eski anlam canlı kalıyor.
- **Deneme süresi ise ortadaki yavaşlığı açıklıyor:** 20'ye indirilince 2.000 etikette %9 yerine %20. Ama sonda yine tekrar çalışma yüzünden düşük kalıyor.
- **Kapı da biraz kayıp bırakıyor:** "ikisinden biri doğru" üst sınırı ile birleşik cevap arasında ~8 puan fark var.

## 3. Öğretmenli senaryo

Son 1.500 kararda modelin kendi verdiği cevaplar:

| | Kendi cevapladığı | Doğruluk |
|---|---|---|
| Düz temel model | 700 | %92,9 |
| Patch'li öğrenci | 891 | %85,0 |

- **"Patch'ler girdiyi tanıdık gösterdi, o yüzden cevap verdi" durumu hiç yok** (0 karar). O hipotez yanlıştı.
- Patch'li öğrencinin 134 yanlışının **131'inde içerideki kalıcı model de yanlış**. Patch'ler yalnızca 30'unda doğru. Kapının bu yanlışlarda patch'lere verdiği ağırlık düşük (0,12).
- **Adil karşılaştırma** (aynı kapsamda hata, test kümesi, 3 tohum, Stage 03–04 sonuçlarından): en emin %80'lik kısımda hata düz temel modelde %17,1, patch'li öğrencide %20,5. AURC 0,090'a karşı 0,115. Yani fark yalnızca "daha çok cevaplıyor" değil; model gerçekten biraz daha zayıf.
- **Olası iki sebep:**
  1. Kalıcı model öğretmenin %20 yanlış etiketlerini tekrar çalışmayla daha çok ezberliyor.
  2. Daha az öğretmene sorduğu için daha az etiket topluyor: 2.750'ye karşı 3.594 öğretmen etiketi.

## Teşhis

Üç zayıflığın ortak kaynağı **patch'ler ya da kapı değil, kalıcı modele yapılan tekrar çalışma.** Tekrar çalışma, yanlış etiketleri ve eski anlamları kalıcı modelde ezberletip canlı tutuyor.

Bu yüzden Stage 05'teki kapı ve güven ayarları işe yaramadı: yanlış parçayı değiştiriyorlardı.

Sıradaki deney: tekrar çalışma kapalıyken tüm senaryolar. Soru şu: sıralı akıştaki kazanç tekrar çalışmadan mı geliyor, yoksa patch'lerden mi?

## Teşhisi sınayan deneyler (tohum 0)

| Tekrar çalışma | Karışık doğruluk | Yanlışı tekrarlama | Drift sonda | Sıralı | Öğretmen: test doğruluğu |
|---|---|---|---|---|---|
| Düz temel model (patch yok) | %88,5 | %5,6 | %67,3 | %12,5 | %74,6 |
| Hepsi (Stage 04) | %89,2 | %20,6 | %65,8 | %80,1 | %73,3 |
| Kapalı | %88,1 | %9,0 | %67,3 | %79,3 | %73,4 |
| Yalnızca doğrulanmış kayıtlar | %88,3 | %7,3 | %65,3 | %79,9 | %72,0 |
| **Kalıcı model ≥ %50 olası buluyorsa** | **%89,1** | **%7,1** | %65,8 | %79,8 | %70,7 |
| Kalıcı model ≥ %20 olası buluyorsa | %89,0 | %9,9 | %66,0 | %80,1 | %70,9 |

Ham çıktılar: `logs/no_rehearsal_seed0.txt`, `logs/rehearse_confirmed_seed0.txt`, `logs/rehearse_if_base_agrees_seed0.txt`.

- Tekrar çalışmayı kapatmak teşhisi doğruluyor: yanlış tekrarlama %20,6'dan %9'a iniyor, drift temel model seviyesine geliyor. Ama karışık akıştaki ~1 puanlık kazanç kayboluyor. O kazanç da tekrar çalışmadan geliyormuş.
- **"Kalıcı model bu cevabı en az %50 olası buluyorsa tekrar çalış"** ikisini birden sağlıyor: yanlış ya da eski etiket, kalıcı modelin geri kalan bilgisiyle çeliştiği için tekrar çalışılmıyor; doğru kayıtlar pekişmeye devam ediyor.

## Seçilen ayarın tam ölçümü (3 tohum)

Commit `07a8bd2`, `results_patch_agree.json`:

| Metrik | Temel model | Stage 04 | **Stage 06 (yeni varsayılan)** |
|---|---|---|---|
| Karışık doğruluk | %88,75 | %89,31 | %89,02 ± 0,23 |
| Karışık, biriken hata (kümülatif log loss) | 0,878 | 0,870 | 0,889 |
| Sıralı doğruluk / forgetting | %13,7 / 0,63 | %80,2 / 0,098 | %80,0 / 0,100 |
| Yanlışı tekrarlama, %1 gürültü | %6,4 | %24,2 | **%11,2 ± 4,2** |
| Yanlışı tekrarlama, %5 gürültü | %6,9 | %21,4 | **%11,2 ± 4,4** |
| Patlama: diğer sınıflara hasar | 3,8 puan | 0,35 puan | 0,58 puan |
| Geri alma süresi | 29,6 sn | 0,12 sn | 0,16 sn |
| Drift yarı ömrü / değişen sınıflar sonda | 3.000 / %64,8 | 3.500 / %61,5 | 3.417 / %61,3 |
| Öğretmen: test doğruluğu / AURC | %73,6 / 0,090 | %71,2 / 0,115 | %70,8 / 0,120 |

**Sonuç**

- **Yanlışı tekrarlama yarıya indi** (%21–24 → %11). Temel modelin seviyesine (%6–7) ise tam inmedi: tohum 0'da %7'ydi, diğer iki tohumda daha yüksek.
- **Bedeli küçük:** karışık akışta −0,3 puan (yine de temel modelden iyi). Biriken hata da temel modelden biraz kötü.
- **Drift ve öğretmenli senaryo değişmedi.** Teşhisin gösterdiği üzere tekrar çalışma bunların sebeplerinden yalnızca biri:
  - **Drift:** asıl yavaşlık deneme süresi gecikmesinden ve patch'lerin eski/yeni anlam arasında bölünmesinden geliyor.
  - **Öğretmen:** model daha az soru sorduğu için daha az öğretmen etiketi topluyor. Bu bir kapsam–risk dengesi ve risk bütçesiyle (Faz 1) ayarlanmalı.
- Bu ayar varsayılan olarak kalıyor.
