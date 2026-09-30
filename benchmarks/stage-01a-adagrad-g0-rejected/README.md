# Aşama 01a — AdaGrad başlangıç birikimi (reddedildi)

**Durum: reddedildi.** Bu aşama, denenip geri alınan bir düzeltmenin ölçümüdür. Neyin neden bırakıldığı görülebilsin diye saklanıyor. Kabul edilen çözüm: [`stage-01b`](../stage-01b-evidence-scaled-updates/).

## Çıkış noktası: ilk kullanıcı denemesinde bulunan hata

Codespaces'te üretilmiş veriyle yapılan denemede:

- Öğretmen "Merhaba, erken rezervasyon yaptırmak istiyorum" için `urgent` sorusuna **P(true) = 0,549** dedi, yani yazı-tura kadar emin değildi.
- Öğrenci bu tek etiketten sonra aynı mesaja **%94** "acil" dedi.
- İlgisiz "Rezervasyonumu iptal etmek istiyorum" mesajı da **%81** acil oldu.
- Aynı durum `department` sorusunda da görüldü: öğretmen %88,6 emindi, bir sonraki cevapta öğrenci %99,6 emindi.

**Kök neden:** lineer uzmanın AdaGrad güncellemesinde yeni bir kelimenin ilk adımı, gradyan ne kadar küçük olursa olsun ±öğrenme hızı büyüklüğünde. Yani etiketin ağırlığı ve belirsizliği normalizasyonda kayboluyor.

## Denenen çözüm

AdaGrad birikimini sıfır yerine `g0`'dan başlatmak; böylece adımlar gradyanla orantılı olur. Ayar taraması (Banking77, 3.000 örnek ve tüm veri, artı hata senaryosu):

| lr / g0 | 3k doğruluk / ECE | Tüm veri doğruluk / ECE | %55'lik etiket sonrası | Yan etki | Tek insan düzeltmesi |
|---|---|---|---|---|---|
| 0,5 / 0 (eski) | %77,4 / 0,013 | %88,5 / 0,013 | %94 | %81 | %97 |
| 0,5 / 0,1 | %74,4 / 0,221 | %87,2 / 0,021 | %15 | %42 | %76 |
| 0,5 / 1,0 | %64,4 / 0,139 | %81,6 / 0,086 | %16 | %35 | %43 |
| 1,0 / 1,0 | %71,7 / 0,237 | %85,6 / 0,013 | %12 | %36 | %68 |
| **2,0 / 1,0 (seçildi)** | %76,3 / 0,031 | %87,9 / 0,015 | %12 | %38 | %97 |
| 1,0 / 0,3 | %75,8 / 0,042 | %88,0 / 0,010 | %13 | %39 | %91 |

## Neden reddedildi

Seçilen ayar taramanın baktığı eksenlerde iyiydi, tam benchmark ise bakılmayan iki eksende kötüleşme gösterdi (commit `f136a38`):

| | Stage 00 | lr 2,0 / g0 1,0 |
|---|---|---|
| 10 örnek/sınıf | %57,2 · ECE 0,122 | %52,4 · **ECE 0,363** |
| 25 örnek/sınıf | %72,8 · ECE 0,019 | %66,7 · ECE 0,070 |
| Tüm veri, karışık | %88,5 · ECE 0,013 | %87,9 · ECE 0,015 |
| Tüm veri, **sıralı akış** | %15,8 | **%1,5** · ECE 0,911 |

Daha büyük öğrenme hızı az veride aşırı güven üretti ve sıralı akıştaki unutmayı ağırlaştırdı. **Ders:** bir ayar taramasında her zaman az veri, kalibrasyon ve sıralı akış eksenleri de ölçülmeli.

## Dosyalar

`banking77_*_online.json` ve `logs/*.txt` — commit `f136a38` ile üretilmiş ham sonuçlar.
