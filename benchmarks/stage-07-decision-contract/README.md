# Aşama 07 — Karar kontratı ve risk bütçesi (Faz 1)

**Amaç:** "%60'tan az eminsen sor" gibi elle seçilen bir eşik yerine, sorunun sahibi hataların ne kadara mal olduğunu söylesin; model cevap verip vermeyeceğine buna göre kendisi karar versin.

- **Kod:** `desic/core/contract.py`, commit `a256833`. Patch'li öğrencinin bütçe ölçümleri `e30779b` ile yapıldı; bu commit yalnızca dashboard'u değiştiriyor.
- **Ortam:** 4 çekirdek, GPU yok. 3 tohum.

## Ne eklendi

Her soruya üç kural eklenebiliyor. Birden fazlası ayarlıysa en sıkısı geçerli. Tanımadığı girdide model yine her zaman soruyor.

| Kural | Ayar | Ne yapar |
|---|---|---|
| Maliyetler | `cost_wrong`, `cost_abstain` | Chow kuralı: (1 − güven) × yanlış maliyeti ≤ sorma maliyeti ise cevaplar. Örnek: 10 ve 1 → %90 eşik. |
| Maliyet matrisi | `cost_matrix` | Hataların maliyeti farklıysa (kötü krediyi onaylamak > iyi krediyi reddetmek), en olası değil **en ucuz** cevabı verir. |
| Risk bütçesi | `risk_budget` | Cevapladıklarının en fazla bu kadarı yanlış olsun. Son etiketli kararlara bakarak, hata oranı %90 güvenle bütçeye sığan en düşük eşiği seçer (Wilson üst sınırı). 30 etiketli karardan önce diğer kurallar karar verir. |

Her karar artık şunlarla birlikte kaydediliyor:
- hangi kuralla verildiği ve kullanılan eşik,
- beklenen maliyet,
- seçilen cevabın olasılığı (*propensity*; şimdilik hep 1),
- politika sürümü.

Bu kayıt, ileride geçmiş kararların farklı bir politikayla nasıl sonuçlanacağını değerlendirebilmek (off-policy) için gerekli. Dashboard'da: Settings → *Decision contract*.

## 1. Bütçe tutuyor mu? (`budget` senaryosu)

Karışık akışta her karar sonradan etiketleniyor. Aynı öğrencide dört kural yan yana çalışıyor. Değerler akışın ikinci yarısından.

"En iyi olası kapsam": o penceredeki etiketleri bilerek seçilebilecek en iyi eşiğin kapsamı.

| Kural | Gerçekleşen hata | Cevapladığı | En iyi olası kapsam | Bütçeyi aşan pencere |
|---|---|---|---|---|
| Bütçe %2 (temel / patch) | %1,3 / %1,4 | %57 / %53 | %65 / %62 | 10 pencerenin 1'i |
| Bütçe %5 | %4,0 / %4,0 | %75 / %74 | %79 / %79 | ~0,3 |
| Bütçe %10 | %8,5 / %8,5 | %89 / %89 | %93 / %93 | 0 |
| Sabit eşik %60 (eski kural) | %7,0 / %6,9 | %86 / %86 | — | — |

- **Bütçe tutuluyor.** Gerçekleşen hata her bütçenin altında kalıyor ve 1.000 kararlık pencerelerin neredeyse hiçbirinde aşılmıyor.
- **Kapsam, geriye dönüp seçilebilecek en iyi eşiğin %86–96'sı.** Temkin payının bedeli bu kadar.
- Eski sabit eşik %7 hatayla %86 kapsıyordu. %10 bütçe aynı güvenlik çerçevesinde %89 kapsıyor. Artık bu denge bir sayı söylenerek seçilebiliyor.

## 2. Öğretmenli senaryo, bütçe %8

Soğuk başlangıç; etiketlerin yalnızca %5'i insandan geliyor.

| | Temel | Patch | Temel + bütçe | Patch + bütçe |
|---|---|---|---|---|
| Öğretmene giden oran (son 500) | %46 | %42 | **%84** | **%93** |
| Kendi cevaplarının doğruluğu | %92,4 | %85,8 | %94,3* | %91,5* |
| Toplam öğretmen çağrısı | 3.594 | 2.910 | 4.778 | 4.910 |
| Test doğruluğu | %73,6 | %70,8 | **%75,3** | **%74,5** |
| Test AURC (düşük iyi) | 0,090 | 0,120 | **0,076** | 0,112 |

\* Bazı tohumlarda son pencerede neredeyse hiç kendi cevabı yok; ortalama yalnızca cevap verdiği tohumlardan.

- **Bütçe burada verdiği sözü fazlasıyla tutuyor, ama çok temkinli.** Etiketli karar az olduğu için (~250 insan etiketi, ve onlar da dağınık) hata oranını %90 güvenle %8'in altında *kanıtlayamıyor*, bu yüzden neredeyse hep soruyor.
- **Yan etki olumlu:** daha çok öğretmen etiketi toplandığı için iki modelin test doğruluğu da yükseliyor. Patch'li öğrencinin Stage 06'daki zayıflığı (%70,8) büyük ölçüde kapanıyor: %74,5.
- **Ama öğretmene bağımlılık hedefi kaçıyor.** Seyrek geri bildirimde yalnızca etikete dayanan bir güvence bu kadar temkinli olmak zorunda.

## Değerlendirme

- **Faz 1'in hedefi sağlandı:** hata ile kapsam arasındaki denge artık bir sayıyla (bütçe ya da maliyet) ayarlanıyor ve verilen söz ölçümde tutuluyor.
- **Açık sorun:** geri bildirim seyrekken risk bütçesi aşırı temkinli. Olası çözüm: etiket azken modelin kendi kalibre olasılıklarına dayanan bir tahmini de kullanmak. Modelin ECE'si 0,03–0,07, yani olasılıkları güvenilir. İki tahminin daha iyimser ama hâlâ güvenli bir birleşimi aranabilir.
- **Kayıtlar artık hazır:** kural, eşik, propensity, politika sürümü. Bu kayıtlar arkadaşının planındaki keşif (exploration) ve off-policy değerlendirme (Faz 4) için gerekli.

## Dosyalar

- `budget_desic.json`, `budget_patch.json`: bütçe senaryosu
- `teacher_desic_budget8.json`, `teacher_patch_budget8.json`: öğretmen senaryosu, bütçe %8
- `logs/`: konsol çıktıları

## Tekrar üretme

```bash
desic eval --scenarios budget --learner desic+patch --seeds 3 --out budget_patch.json
desic eval --scenarios teacher --learner desic+patch --risk-budget 0.08 --seeds 3 --out teacher_patch_budget8.json
```
