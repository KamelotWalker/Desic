# Aşama 11 — Politika/model ayrımı (P1) ve servis edilen aksiyonun riski (S1)

- **Kod:** P1 commit `73f6e79` (`desic/core/policy.py`); S1 commit `f409a56`.
- **Ölçüm:** öğretmenli senaryo, 3 tohum, commit `f409a56`.

## P1: karar politikası modelden ayrı

- `DecisionPolicy` nesnesi eşik, risk bütçesi, bütçe modu ve maliyetleri tutuyor, kendi sürümüyle. Model sürümü yalnızca model öğrendiğinde ya da geri aldığında artıyor.
- Her karar `policy_version` ve `model_version`'ı ayrı ayrı logluyor.
- Politika değişikliği modelin cevabını değiştirmiyor; öğrenme de politikayı değiştirmiyor. Kanıt: `tests/test_policy.py`.
- **Aday politika yeniden oynatma:** `POST /v1/questions/{name}/policy/replay`, dashboard'da *Preview on past decisions*. Geri bildirim almış geçmiş kararlar, o an servis edilen olasılıklarla yeni bir politikaya göre yeniden değerlendiriliyor: kapsam, hata, maliyet. Model değişmiyor.

## S1: kullanıcıya gerçekten verilen cevabın hatası

Geri bildirim artık kullanıcının *aldığı* cevabı ve onu kimin ürettiğini (öğrenci, öğretmen, kural) puanlıyor. Bu kayıt geri almayla birlikte siliniyor. Dashboard'daki karşılığı: *Served error*.

Öğretmenli senaryo, kullanıcıya verilen cevabın hatası (öğrenci cevapladıysa öğrencinin, sorduysa öğretmenin cevabı; öğretmen ~%80 doğru):

| | Öğrencinin kendi cevap doğruluğu (son 500) | Öğretmene giden | **Kullanıcının gördüğü hata, 2. yarı** | Test doğruluğu |
|---|---|---|---|---|
| Temel model, bütçesiz | %92,4 | %46 | **%14,9** ± 0,7 | %73,6 |
| Patch'li, bütçesiz | %85,8 | %42 | %18,3 ± 2,2 | %70,8 |
| Patch'li, beklenen %8 | %95,0 | %71 | %15,9 ± 1,0 | %74,4 |
| Patch'li, **garantili %8** | %94,4 | %95 | **%19,1** ± 0,5 | %75,2 |

**Bulgu: öğrenci üzerinde tanımlı risk bütçesi, kullanıcının gördüğü riski yanlış temsil ediyor.**

- Garantili %8 bütçe, öğrencinin hatasını mükemmel biçimde düşük tutuyor (%94 doğru). Ama bunu kararların %95'ini %80 doğru bir öğretmene göndererek yapıyor. Sonuçta kullanıcı **en yüksek hatayı** (%19) görüyor.
- Çekimser kalmak bedava değil: sorulan tarafın hatası, kullanıcının hatası oluyor.
- "Beklenen" mod bu yüzden daha iyi (%15,9). En düşük hatayı ise hâlâ bütçesiz temel model veriyor (%14,9).

## Sonraki hipotez (S2)

Politika, yönlendirmenin de hatalı olabileceğini hesaba katmalı. Öğretmenin doğruluğu servis edilen aksiyon kayıtlarından ölçülebiliyor (`served_actions.by_source.teacher`). Öğrencinin bu doğruluktan daha emin olduğu bir kararı öğretmene göndermek kullanıcının hatasını artırır. Bu yüzden:

- çekimserlik eşiği, ölçülen öğretmen doğruluğunu geçmemeli,
- risk bütçesi öğrenci cevabı yerine servis edilen aksiyon üzerinden tanımlanabilmeli.

Kabul ölçütü: öğretmenli senaryoda kullanıcının gördüğü hata, bütçesiz temel modelin %14,9'undan düşük; test doğruluğu korunuyor.

## Dosyalar

- `teacher_desic.json`, `teacher_patch.json`, `teacher_patch_expected8.json`, `teacher_patch_guaranteed8.json`
- `logs/`: konsol çıktıları
