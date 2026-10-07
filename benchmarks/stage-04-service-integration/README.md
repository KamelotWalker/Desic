# Aşama 04 — Servise entegrasyon ve uyarlanır deneme süresi

**Kısaca:** Stage 03'teki geri alınabilir katman (bir tür "geçici not defteri") artık dashboard'da ve API'de kullanılıyor. Entegrasyon sırasında bulunan iki sorun düzeltildi. Değişen ayarla ölçüm tekrarlandı.

- **Kod:**
  - Ölçüm: commit `e81fc83`.
  - Disk boyutu düzeltmesi: commit `be7395a`. Bu düzeltme modelin kararlarını değiştirmez.
- **Ortam:** Python 3.11, 4 çekirdek, GPU yok. 7 senaryo × 3 tohum.
- **Karşılaştırma:** Stage 03'teki temel model ve Stage 03'teki patch katmanı, birebir aynı akışlar üzerinde.

## Ne değişti

**1. Servis ve dashboard**
- Her soru artık patch katmanından geçiyor. Önceden kaydedilmiş sorular yüklenirken sarılıyor; metrik geçmişleri korunuyor.
- Feedback log'daki **Retract** düğmesi etiketi anında geri alıyor; artık ayrıca "rebuild" gerekmiyor. **Restore** etiketi yeniden öğretiyor.
- **"Undo last N"** ile son N etiket tek seferde geri alınabiliyor, istenirse kaynağa göre (insan / öğretmen / veri seti). API karşılığı: `POST /v1/questions/{name}/retract-recent`.
- Her etiket önce kayda yazılıyor; kayıt numarası, o etiketin geri alma anahtarı oluyor.
- Açıklamalarda cevabın dayandığı en yakın örnekler gösteriliyor: benzerlik, güven ve "hâlâ deneme süresinde mi".

**2. Uyarlanır deneme süresi (bulunan sorun 1)**
- Stage 03'te deneme süresi sabit 500 etiketti. 300 etiketli bir soruda kalıcı model hiçbir şey öğrenmemiş oluyordu: açıklamadaki "kanıt kelimeler" kayboluyordu ve her şey yalnızca benzer örneklere dayanıyordu.
- Artık deneme süresi **görülen etiketlerin %10'u**; en az 20, en fazla 500.
- Küçük bir soruda eski bir etiketi geri almak zaten ucuz, çünkü yalnızca birkaç yüz etiket yeniden oynatılıyor.

**3. Disk boyutu (bulunan sorun 2)**
- 10 bin etiketli bir soru diske **243 MB** yazılıyordu: kalıcı modelin 4 tam kopyası, geri alma kontrol noktası olarak saklanıyordu.
- Kontrol noktaları artık yalnızca bellekte tutuluyor. Boyut **62 MB**; temel model 52 MB.
- Bunun bedeli: sunucu yeniden başladıktan sonra çok eski bir etiketi geri almak, yeni kontrol noktaları oluşana kadar baştan yeniden oynatma gerektiriyor.
- Ayrıca otomatik snapshot'ların yalnızca son 10'u tutuluyor. Bu sorun patch katmanından önce de vardı: 10 bin etiketlik bir soru 40 snapshot × 52 MB = ~2 GB yer kaplıyordu.

## Sonuçlar (3 tohum, ortalama ± std)

| Senaryo | Metrik | Temel model | Stage 03 (sabit 500) | **Stage 04 (uyarlanır)** |
|---|---|---|---|---|
| Karışık | Doğruluk | %88,75 | %88,86 | **%89,31 ± 0,10** |
| | Log loss | 0,423 | 0,410 | **0,398** |
| | Akış boyunca biriken hata (kümülatif log loss) | 0,878 | 0,979 ✗ | **0,870** ✓ |
| | ECE | 0,015 | 0,013 | 0,015 |
| Sıralı | Doğruluk | %13,7 | %80,2 | **%80,2** |
| | Forgetting index | 0,63 | 0,097 | 0,098 |
| %1 / %5 gürültü | Doğruluk | %88,5 / %87,9 | %88,7 / %88,0 | **%89,1 / %88,4** |
| | Yanlışı tekrarlama | %6,4 / %6,9 | %22,8 / %23,4 | %24,2 / %21,4 ✗ |
| Patlama | Kurban, hemen sonra | %0,8 | %65 | %65 |
| | Diğer sınıflardaki hasar | 3,8 puan | 0,5 puan | **0,35 puan** |
| | Geri alma süresi / yeniden oynatılan olay | 29,6 sn / 5.001 | 0,105 sn / 0 | 0,117 sn / 0 |
| Drift | Toparlanma yarı ömrü | 3.000 | 3.417 ✗ | 3.500 ✗ |
| | Değişen sınıflar, sonda | %64,8 | %59,9 | %61,5 |
| Öğretmen | Öğretmene giden oran (son 500) | %46,3 | %41,1 | **%35,3** |
| | Kendi cevaplarının doğruluğu | %92,4 | %88,9 | %85,6 ✗ |
| | Test doğruluğu / ECE | %73,6 / 0,115 | %71,8 / 0,046 | %71,2 / **0,035** |

## Kontrol deneyi: kazanç sadece "daha büyük hafıza"dan mı geliyor?

Temel modelin hafızası yalnızca son 2.000 örneği tutuyordu. Kontrol olarak hafızayı 20.000'e çıkardım; böylece hiçbir örnek atılmıyor (`control_base_big_memory.py`, tohum 0):

| Sıralı akış | Doğruluk | Forgetting index |
|---|---|---|
| Temel model | %13,7 | 0,63 |
| Temel model + 10× hafıza | **%12,7** | 0,65 |
| Patch katmanı | %80,2 | 0,10 |

Hafızayı büyütmek işe yaramıyor. Sebep: Hedge karışımı hafıza uzmanına çok az ağırlık veriyor (%1–2). Bu yüzden onun bildiğinin cevaba etkisi neredeyse yok.

Kazanç, patch katmanının **öğrenilen yerel kapısından** geliyor. Kapı, unutan lineer modelin yerine benzer örneklerin sözünü geçirmeyi bağlam bazında öğreniyor.

## Değerlendirme

- **Deneme süresini uyarlanır yapmak, Stage 03'ün bir zayıflığını tamamen düzeltti.** Akış boyunca biriken hata artık temel modelden iyi (0,870'e karşı 0,878). Karışık akışta doğruluk da arttı: %88,75'ten %89,31'e. Sebep: kalıcı model yeni etiketleri artık daha erken öğreniyor.
- **Hâlâ kötü olanlar:**
  - Drift'e uyum daha yavaş.
  - Aynı mesajda yanlış etiketi tekrarlama oranı yüksek.
  - Öğretmenli senaryoda model daha çok kendisi cevaplıyor ama daha az doğru.

  Bu üçünün ortak sebebi büyük olasılıkla aynı: katman yerel etiketlere hızlı güveniyor ve eski/yanlış yerel kanıtı yeterince hızlı unutmuyor. Sonraki adayların hedefi bu.
- **Maliyet:** yaklaşık 2 kat yavaş. Bu koşu 3 paralel işçiyle yapıldı ve aynı anda başka işler de çalışıyordu; o yüzden toplam süre Stage 03 ile doğrudan karşılaştırılamaz.

## Dosyalar

- `results_patch.json`: patch katmanı, uyarlanır deneme süresi (commit `e81fc83`)
- `logs/eval_patch.txt`: konsol çıktısı
- `control_base_big_memory.py`, `logs/control_base_big_memory.txt`: kontrol deneyi

## Tekrar üretme

```bash
desic eval --learner desic+patch --seeds 3 --out results_patch.json
python benchmarks/stage-04-service-integration/control_base_big_memory.py
```
