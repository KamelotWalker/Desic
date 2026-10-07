# Aşama 12 — Yönlendirmeyi hesaba katan politika (S2)

**Hipotez** ([RESEARCH.md](../../RESEARCH.md), S2): Politika öğretmenin ölçülen doğruluğunu hesaba katarsa, öğretmenli senaryoda kullanıcının gördüğü hata, bütçesiz temel modelin %14,9'unun altına iner.

**Sonuç: kabul ölçütü sağlanmadı.** Mekanizma patch'li öğrencinin kullanıcı hatasını 2,6 puan düşürüyor ve iki öğrencinin test doğruluğunu ~2,5 puan artırıyor, ama %14,9'un altına inemiyor. `escalation_aware` ayarı **isteğe bağlı** olarak kalıyor; varsayılanı kapalı.

- **Kod:** `desic/core/policy.py`. Son sürüm commit `b0e01aa`.
- **Ölçüm:** öğretmenli senaryo, 3 tohum. Öğretmen kalibre ve ~%80 doğru; insan etiketi kararların %5'i.

## Üç deneme

| # | Kural | Neden değiştirildi |
|---|---|---|
| 1 | Öğrenci, ölçülen öğretmen doğruluğu kadar eminse çekimser kalmaz | Bütçesiz ayarda hiçbir şeyi değiştirmedi: eşik %60, öğretmen %80; model zaten yalnızca öğretmenden kötü olduğu yerde soruyordu |
| 2 | + yalnızca elle konan eşik varken, öğretmenden *az* eminse sorar (iki yönlü) | Temel model az kendinden emin (ECE 0,12): "%70" dediği cevaplar çok daha sık doğru. Ham güvenle kıyas onu gereksiz yere öğretmene gönderdi (%14,9 → %15,5) |
| 3 | + güven yerine, öğrencinin **o güven seviyesindeki ölçülmüş doğruluğu** kullanılır (±0,1 komşuluk, az veride güvene doğru büzülür) | Son sürüm |

## Sonuçlar (son sürüm, 3 tohum)

| | Kullanıcının gördüğü hata, 2. yarı | Öğretmene giden (son 500) | Kendi cevap doğruluğu | Test doğruluğu | Öğretmen çağrısı |
|---|---|---|---|---|---|
| Temel model | %14,9 ± 0,7 | %46 | %92,4 | %73,6 | 3.594 |
| Temel model + S2 | %15,0 ± 1,5 | %58 | %95,4 | **%76,4** | 4.129 |
| Patch'li | %18,3 ± 2,2 | %42 | %85,8 | %70,8 | 2.910 |
| Patch'li + S2 | **%15,7** ± 0,9 | %57 | %91,5 | **%73,4** | 3.760 |
| Patch'li + beklenen %8 + S2 | %16,3 ± 0,9 | %59 | %93,4 | %73,4 | 4.040 |
| (Stage 11) Patch'li + garantili %8 | %19,1 | %95 | %94,4 | %75,2 | 4.949 |
| (deneme 1) Patch'li + garantili %8 + S2 | %16,0 | %56 | %91,5 | %72,6 | 3.622 |

## Neden %14,9'un altına inilemiyor

- Kullanıcının gördüğü hatanın bir tabanı var: tanımadığı girdiler her zaman öğretmene gidiyor. Öğretmen de %20 yanılıyor.
- İkinci yarıda kararların ~%45–57'si öğretmene gidiyor. Bu tek başına ~%9–11 hata demek.
- Bu tabanı politika değil, öğrencinin **bilgisi** belirliyor: kaç girdiyi tanıyıp doğru cevaplayabildiği.
- S2 bu tabana yaklaştırıyor; altına inmek daha iyi bir öğrenci gerektirir (örneğin önceden eğitilmiş encoder).

## Değerlendirme

- **Hipotez reddedildi.** Mekanizma yine de değerli: patch'li öğrencide kullanıcı hatası −2,6 puan, test doğruluğu iki öğrencide de +2,5–2,8 puan. Öğretmen etiketleri, öğrencinin gerçekten zayıf olduğu yerde toplanıyor.
- **Bedeli:** %15–30 daha fazla öğretmen çağrısı. Öğretmen bir LLM API'si olduğunda bu para demek. Bu yüzden varsayılan değil; ayar olarak sunuluyor.
- **Genel ders:** "Kendi doğruluğunu ölç, başkasının doğruluğunu ölç, iyi olana yönlendir". Bu, ham güvenden değil **ölçülmüş** doğruluktan yapılmalı. Ham güvenle yapılınca kalibrasyonu bozuk öğrenci yanlış yönlendirildi (deneme 2).

## Dosyalar

- `teacher_*_aware3.json`: son sürüm
- `teacher_*_aware2.json`: deneme 2
- `teacher_*_aware.json`: deneme 1
- `teacher_patch_rerun.json`: karşılaştırma
- `logs/`
