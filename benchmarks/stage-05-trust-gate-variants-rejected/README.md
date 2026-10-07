# Aşama 05 — Güven ve kapı varyantları (reddedildi)

**Durum: reddedildi.** Hiçbir varyant hedeflenen üç zayıflığı belirgin şekilde düzeltmedi. Bu yüzden varsayılan ayarlar değişmedi. Neyin denendiği görülsün diye saklanıyor.

- **Kod:** commit `5bfdf0d` (ayarlar eklendi, varsayılanları kapalı). Ölçüm tek tohumla (0), tam Banking77 verisiyle yapıldı.
- **Hedef:** Stage 04'te kalan üç zayıflık:
  - drift'e geç uyum,
  - aynı mesajda yanlış etiketi tekrarlama,
  - öğretmenli senaryoda modelin kendi cevaplarının daha az doğru olması.

## Denenen fikirler

| Varyant | Fikir | Beklenen etki |
|---|---|---|
| `gate-gt` | Kapı (patch'e mi kalıcı modele mi güvenileceğine karar veren kısım) yalnızca kesin etiketlerden öğrensin, öğretmenin (%80 doğru) etiketlerinden öğrenmesin | Öğretmenli senaryoda doğruluk artar |
| `trust-fade` | Bir patch'in destek/itiraz kanıtı zamanla sönsün (yarı ömür 1.000 etiket); güveni %50'nin altına düşen patch artık tekrar çalıştırılmasın | Drift'e daha hızlı uyum |
| `gate-prior` | Hiç görülmemiş bir bağlamda kapı, kalıcı modele daha çok güvenerek başlasın (patch ağırlığı 0,5 yerine 0,2) | Yanlışı daha az tekrarlama |
| `all` | Hepsi birden | — |

## Sonuçlar (tohum 0)

| Metrik | Stage 04 ayarı | gate-gt | trust-fade | gate-prior | all |
|---|---|---|---|---|---|
| Drift yarı ömrü (düşük iyi) | 3.750 | 3.750 | 3.500 | 3.750 | 3.500 |
| Drift: değişen sınıflar, sonda | %65,8 | %65,8 | %62,5 | %65,8 | %62,5 |
| Yanlışı tekrarlama (%5 gürültü) | %20,6 | %20,6 | %22,7 | %20,3 | %22,3 |
| Öğretmen: kendi cevap doğruluğu | %87,0 | %84,0 | %86,8 | %85,1 | %88,4 |
| Öğretmen: öğretmene giden oran | %35,2 | %36,2 | %43,8 | %46,4 | %39,8 |
| Öğretmen: test doğruluğu | %73,3 | %72,6 | %74,5 | %73,1 | %72,6 |
| Sıralı akış doğruluğu | %80,1 | %80,1 | %80,9 | %80,1 | %80,9 |
| Karışık akış doğruluğu | %89,2 | %89,2 | %89,0 | %89,2 | %89,0 |
| Patlama: diğer sınıflara hasar | 0,03 puan | 0,03 | 0,24 | 0,03 | 0,24 |

## Ne öğrendik

- **`gate-gt` ve `gate-prior` neredeyse hiçbir şeyi değiştirmedi.** Kapının nasıl başladığı ve kimden öğrendiği bu üç zayıflığın sebebi değil.
- **`trust-fade`** drift'te kaybın yarısını biraz daha erken geri kazanıyor (3.750 → 3.500), ama akışın sonunda değişen sınıflarda daha kötü (%65,8 → %62,5) ve yanlışı daha çok tekrarlıyor. Net kazanç yok.
- **Öğretmen senaryosundaki farklar** (%84–88) tek tohumda gürültü seviyesinde.
- **Sonuç:** üç zayıflığın sebebi hakkındaki tahmin (kapı ya da güven mekanizması) büyük ölçüde yanlıştı. Bir sonraki adım tahmin değil, ölçerek teşhis: her zayıflıkta hatanın hangi parçadan (kalıcı model, patch'ler, kapı, kalibrasyon) geldiğini ayrı ayrı ölçmek.

## Tekrar üretme

```bash
python benchmarks/stage-05-trust-gate-variants-rejected/sweep.py "$(cat benchmarks/stage-05-trust-gate-variants-rejected/variants.json)"
```
