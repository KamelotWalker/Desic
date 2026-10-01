# Aşama 09 — Drift: eskimiş patch'leri oydan çıkarmak (reddedildi)

**Durum: reddedildi.** Drift'e uyum hiç değişmedi; varsayılanlar değişmedi.

- **Kod:** `PatchStore(supersede_below=…)`, commit `3c8040d`. Varsayılan olarak kapalı.
- **Ölçüm:** tek tohum (0), tam Banking77.
- **Script:** `benchmarks/stage-05-trust-gate-variants-rejected/sweep.py`.

## Fikir

Stage 06 teşhisi, drift sırasında patch oylarının %28–45'inin eski anlamdan geldiğini göstermişti. Deneme şuydu:

- Sonradan gelen benzer etiketlerle çürütülen kaydın güveni zamanla sönerek hesaplansın (yarı ömür 500 ya da 1.000 etiket).
- Güveni eşiğin (0,35 ya da 0,5) altına düşen kayıt artık hiç oy vermesin.

## Sonuçlar (tohum 0)

| Varyant | Drift yarı ömrü | Değişen sınıflar, sonda | Sıralı | Yanlış tekrarlama | Karışık | Patlama hasarı |
|---|---|---|---|---|---|---|
| Stage 06 varsayılanı | 3.750 | %65,8 | %79,8 | %7,1 | %89,1 | 0,17 puan |
| yarı ömür 500, eşik 0,35 | 3.750 | %65,8 | %80,7 | %8,1 | %88,8 | 0,00 |
| yarı ömür 1.000, eşik 0,35 | 3.750 | %65,8 | %80,7 | %7,5 | %88,8 | 0,26 |
| yarı ömür 500, eşik 0,50 | 3.750 | %65,8 | %80,7 | %7,1 | %88,8 | 0,07 |

## Ne öğrendik

- **Drift sonuçları birebir aynı.** Patch oylarını temizlemek drift'te cevabı değiştirmiyor. Stage 06'ya göre bunun sebebi, drift sırasında kapının zaten büyük ölçüde kalıcı modele güvenmesi.
- **Drift'i sınırlayan kalıcı model:**
  - Yeni anlamı deneme süresi kadar (büyük soruda 500 etiket) geç öğreniyor.
  - Eski anlamı silmesi AdaGrad'ın zamanla küçülen öğrenme hızı yüzünden yavaş.

  Stage 06'da deneme süresini 20'ye indirmek ortadaki yavaşlığı azaltmıştı, ama sonucu temel modele eşitlememişti.
- **Burada bir denge var:** deneme süresi, anında ve birebir geri alınabilen pencere. Kısaltılırsa drift'e daha hızlı uyulur, ama yanlış etiketler daha çabuk kalıcı modele girer. Patlama senaryosu, drift senaryosunun ayna görüntüsü: model için ikisi de "bir bölgede etiketlerin anlamı değişti" demek. Fark yalnızca hangisinin doğru olduğunda.
- **Drift için asıl iş kalıcı modelin kendisinde:** örneğin yerel bir değişim tespit edildiğinde o bölgede öğrenme hızını artırmak. Bu, patch katmanından bağımsız ayrı bir çalışma.
