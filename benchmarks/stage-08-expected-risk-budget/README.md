# Aşama 08 — "Beklenen" risk bütçesi modu

**Amaç:** Stage 07'deki risk bütçesi, geri bildirim seyrekken aşırı temkinliydi. Hata oranını yalnızca etiketlerle kanıtlamaya çalıştığı için öğretmenli senaryoda kararların %93'ünü öğretmene gönderiyordu. Bu aşamada ikinci bir mod ekleniyor.

- **Kod:** `desic/core/contract.py` (`expected_threshold`), commit `cff5038`.
- **Ortam:** 4 çekirdek, GPU yok. 3 tohum.

## Yöntem

`risk_budget_mode = "expected"` modunda bir τ eşiğinin üstündeki kararların hata oranı şöyle tahmin ediliyor. Bu bir *prediction-powered* tahmin (Angelopoulos ve ark., 2023):

    tahmini hata(τ) = ortalama(1 − güven)                 son kararların tümünde, güven ≥ τ (etiketli ya da değil)
                    + ortalama(hata − (1 − güven))        yalnızca etiketli olanlarda, güven ≥ τ (en az 10 tane varsa)

- **Birinci terim:** modelin kendi kalibre güveninden gelen tahmin.
- **İkinci terim:** etiketlerin gösterdiği kalibrasyon sapması.

Seçilen eşik, tahmini hatanın bütçeye sığdığı en düşük eşik.

Bunun için kararların güveni artık kaydediliyor (`TaskMetrics.served`). Bütçe güvenlik payıyla değil, **ortalamada** tutuluyor.

## 1. Bol geri bildirim (`budget` senaryosu, patch'li öğrenci)

| Bütçe | Garantili: hata / kapsam | **Beklenen: hata / kapsam** | En iyi olası kapsam | Bütçeyi aşan pencere (10'da), garantili / beklenen |
|---|---|---|---|---|
| %2 | %1,4 / %53 | %2,2 / %62 | %62 | 1 / 5 |
| %5 | %4,0 / %74 | %5,0 / %79 | %79 | 0,3 / 5 |
| %10 | %8,5 / %89 | %9,7 / %92 | %93 | 0 / 2 |

- **Beklenen mod**, geriye dönüp seçilebilecek en iyi eşikle neredeyse aynı kapsama ulaşıyor (%98–100). Gerçekleşen hata bütçenin tam üstünde ya da altında; pencerelerin yaklaşık yarısında bütçe biraz aşılıyor. "Ortalamada" sözünün anlamı bu.
- **Garantili mod** hiç aşmıyor, ama bunun bedeli kapsamın %4–15'i.

## 2. Seyrek geri bildirim (öğretmenli senaryo, bütçe %8)

| | Bütçe yok | Garantili %8 | **Beklenen %8** |
|---|---|---|---|
| **Patch'li öğrenci** | | | |
| Öğretmene giden oran (son 500) | %42 | %93 | **%71** |
| Kendi cevaplarının doğruluğu (son 500) | %85,8 | %91,5* | **%95,0** |
| Test doğruluğu | %70,8 | %74,5 | **%74,4** |
| Test AURC (düşük iyi) | 0,120 | 0,112 | **0,095** |
| **Temel model** | | | |
| Öğretmene giden oran | %46 | %84 | **%40** |
| Kendi cevaplarının doğruluğu | %92,4 | %94,3* | %90,1 |
| Test doğruluğu | %73,6 | %75,3 | %74,0 |

\* Bazı tohumlarda son pencerede neredeyse hiç kendi cevabı yok.

- **Patch'li öğrenci + beklenen mod en iyi dengeyi veriyor.** Kendi cevapları %95 doğru, yani %8 bütçenin rahatça içinde. Öğretmene gidenler %93'ten %71'e iniyor. Test doğruluğu, garantili moddaki kadar yüksek (%74,4).
- **Patch'li öğrencinin Stage 06'da kalan öğretmenli senaryo zayıflığı kapandı:** test doğruluğu %70,8'den %74,4'e çıktı (temel model: %73,6). AURC 0,120'den 0,095'e indi.
- Temel modelde beklenen mod öğretmene giden oranı düşürüyor (%40). Kendi cevaplarındaki hata %9,9, yani bütçenin biraz üstünde. Ortalamada tutan bir mod için beklenebilir bir sapma.

## Değerlendirme

- İki mod iki farklı ihtiyaç için:
  - **Garantili:** yanlış cevabın pahalı olduğu ve etiketin bol olduğu durumlar.
  - **Beklenen:** etiketin az olduğu ve maliyetin ortalamada önemli olduğu durumlar.
- Dashboard'da mod, *Settings → Decision contract* altından seçiliyor. Varsayılan hâlâ *garantili*.
- Faz 1 (karar kontratı) bu aşamayla tamamlandı.

## Dosyalar

- `budget_patch.json`: bol geri bildirim, iki mod
- `teacher_patch_expected8.json`, `teacher_desic_expected8.json`: öğretmenli senaryo, beklenen %8
- Karşılaştırma için bütçesiz ve garantili sonuçlar: Stage 06 ve Stage 07 klasörleri
- `logs/`: konsol çıktıları

## Tekrar üretme

```bash
desic eval --scenarios budget --learner desic+patch --seeds 3 --out budget_patch.json
desic eval --scenarios teacher --learner desic+patch --risk-budget 0.08 --risk-budget-mode expected --seeds 3 --out teacher_patch_expected8.json
```
