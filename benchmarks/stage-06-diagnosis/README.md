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
