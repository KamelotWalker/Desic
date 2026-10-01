# Desic — araştırma tezi ve kanıt durumu

Desic bir "çevrimiçi öğrenen model" olarak değil, **geri alınabilir, sürekli öğrenen bir karar runtime'ı** olarak ilerliyor. Tez dört özellikten oluşuyor:

1. **Immediate adaptation:** bir geri bildirim bir sonraki kararı hemen etkiler, drift ve seyrek geri bildirim altında da bozulmadan.
2. **Bounded learning blast radius:** bir öğrenme olayının etkisi sınırlı, ölçülebilir ve izlenebilir bir bölgede kalır.
3. **Explicit decision risk:** maliyet ve risk, gerçekten *servis edilen aksiyon* üzerinden tanımlanır, kontrol edilir ve ölçülür. Karar politikası model durumundan ayrıdır.
4. **Reversible state evolution:** her öğrenme olayı, etkilediği tüm durumla birlikte geri alınabilir.

**Çalışma disiplini.** Her değişiklik bir hipotezle başlar. Hata modu izole edilir, ablasyonla doğrulanır, sonuç kötüyse çöpe atılır. Reddedilen denemeler de `benchmarks/` altında kalır (01a, 05, 09).

## Kanıt tablosu

Değerler, aynı akışlarda temel modele ([aşama 03](benchmarks/stage-03-reversible-patches/)) karşı 3 tohum ortalamasıdır; aksi belirtilmedikçe Banking77 kullanılır.

| Özellik | Ölçüt | Temel model | Desic (güncel) | Durum | Kaynak |
|---|---|---|---|---|---|
| Immediate adaptation | Tek düzeltmenin aynı girdide hemen tutması | ✓ | ✓ | ✅ | `tests/test_api.py` |
| | Sıralı akış (class-incremental) doğruluğu / forgetting | %13,7 / 0,63 | %80,0 / 0,10 | ✅ | [06](benchmarks/stage-06-diagnosis/) |
| | Drift yarı ömrü (düşük iyi) | 3.000 | 3.417 | ❌ temel modelden yavaş | [06](benchmarks/stage-06-diagnosis/), [09](benchmarks/stage-09-supersede-rejected/) |
| | Seyrek geri bildirim: kendi cevap doğruluğu / öğretmene gidenler (bütçe %8, beklenen mod) | %92,4 / %46 (bütçesiz) | %95,0 / %71 | ✅ | [08](benchmarks/stage-08-expected-risk-budget/) |
| Bounded blast radius | 30 hatalı etiketin diğer sınıflara hasarı, *hemen sonra* | 3,8 puan | 0,6 puan | ✅ | [06](benchmarks/stage-06-diagnosis/) |
| | Aynı hasar, ikize göre: hemen / +600 (pekiştirme sonrası) / +1.000 | 3,8 / 0,1 / 0,0 puan | 0,5 / **1,0** / 0,1 puan | 🟡 küçük ve geçici, ama pekiştirmede yayılıyor | [10](benchmarks/stage-10-blast-radius/) |
| | Yanlış etiketi aynı girdide tekrarlama | %6,9 | %11,2 | 🟡 | [06](benchmarks/stage-06-diagnosis/) |
| Explicit decision risk | Risk bütçesi, servis edilen öğrenci cevabında (%2 / %5 / %10) | — | %1,4 / %4,0 / %8,5 (garantili) | ✅ | [07](benchmarks/stage-07-decision-contract/) |
| | Kapsam / en iyi olası kapsam | — | %86–96 (garantili), %98–100 (beklenen) | ✅ | [08](benchmarks/stage-08-expected-risk-budget/) |
| | Öğretmen ve kural cevaplarının da risk hesabına girmesi | — | **girmiyor** (yalnızca öğrencinin kendi cevabı sayılıyor) | ⚠️ açık | — |
| | Maliyet matrisi (asimetrik hata) senaryosu | — | **ölçülmedi** | ⚠️ açık | — |
| | Politika ile model durumunun ayrılması | — | Kısmi: kural ve eşik loglanıyor, ama "politika sürümü" = model sürümü | ⚠️ açık | — |
| Reversible state | Deneme süresindeki etiketi geri alma | 29,6 sn, 5.001 olay | 0,16 sn, 0 olay, birebir | ✅ | [06](benchmarks/stage-06-diagnosis/), `tests/test_patches.py` |
| | Pekiştirilmiş etiketi geri alma | tam yeniden oynatma | kontrol noktasından yeniden oynatma, birebir | ✅ | `tests/test_patches.py` |
| | Geri alınan etiketin risk kontrolünden (servis metrikleri) silinmesi | — | siliniyor; risk eşiği, o etiketi hiç görmemiş bir sistemle birebir aynı (R1) | ✅ | `tests/test_patches.py::test_retracted_labels_leave_the_risk_controller_too` |
| | Yeniden başlatmadan sonra pekiştirilmiş etiketi geri alma | — | baştan yeniden oynatma (kontrol noktaları diske yazılmıyor) | 🟡 | [04](benchmarks/stage-04-service-integration/) |

## Açıklar ve sıradaki hipotezler

| # | Hipotez | Hata modu / ölçüm | Kabul ölçütü |
|---|---|---|---|
| R1 ✅ | Geri alınan bir etiket, risk kontrolünün kullandığı servis metriklerinden de çıkarılırsa, geri almadan sonraki risk eşiği o etiketi hiç görmemiş bir sistemle aynı olur. | Şu an geri alınan yanlış etiketler eşiği etkilemeye devam ediyor. | Unit test: geri alma sonrası eşik = karşı-olgusal eşik. |
| B1 🟡 | 30 hatalı etiket deneme süresini geçip pekiştirilince, diğer sınıflara hasar temel modelin seviyesine (3,8 puan) geri döner. | `burst` senaryosuna "pekiştirmeden sonra" ölçümü eklenecek. | Doğrulanırsa: pekiştirme kapısı (çelişkili bölgede kanıt toplanana kadar beklet) bir ablasyonla denenecek. |
| P1 | Karar politikası (kontrat, eşikler, risk kontrolcüsü) model durumundan ayrı bir nesne ve sürüm olursa, politika değişikliği modele dokunmadan uygulanır ve geri alınır. Loglanan kararlar yeni bir politikayla yeniden değerlendirilebilir. | Şu an politika sürümü = model sürümü; ayar değişikliği ayrı izlenmiyor. | Her kararda (model sürümü, politika sürümü) ayrı loglanıyor; politika değişikliği modeli değiştirmiyor (test). |
| S1 | Servis edilen *son* aksiyon (öğretmen ya da kural cevabı dahil) risk hesabına girerse, raporlanan risk kullanıcının gerçekten gördüğü hatayı yansıtır. | Şu an yalnızca öğrencinin kendi cevabı sayılıyor. | Öğretmenli senaryoda "servis edilen aksiyon hatası" ayrı raporlanıyor. |
| C1 | Maliyet matrisi, asimetrik maliyetli bir görevde toplam maliyeti en olası cevabı seçmekten daha düşük tutar. | Ölçülmedi. | Kredi demosunda maliyet matrisli ve matrissiz toplam maliyet karşılaştırması. |
| D1 | Drift, patch katmanıyla değil kalıcı modelin uyum hızıyla sınırlı ([09](benchmarks/stage-09-supersede-rejected/)). Yerel çelişki yoğunluğu yüksek bölgelerde pekiştirmenin hızlandırılması yarı ömrü kısaltır. Ama aynı mekanizma hatalı etiket patlamasını da hızlandırır. | Drift ile patlama, model için aynı sinyal. | İkisi birlikte raporlanacak; ikisini birden iyileştirmeyen sürüm kabul edilmez. |

**B1 sonucu:** hasar pekiştirmede 0,5'ten 1,0 puana çıkıyor, sonra iyileşiyor. Temel modelin 3,8'ine dönmüyor. Pekiştirmede sönümleme hasarı sıfırlıyor, ama drift'i (%65,8 → %50) ve karışık doğruluğu (−3 puan) bozuyor: reddedildi ([10](benchmarks/stage-10-blast-radius/)).

**Yeni hipotez:**

| # | Hipotez | Hata modu / ölçüm | Kabul ölçütü |
|---|---|---|---|
| K1 | Etiketin *kaynağına* göre bir güven puanı (kaynağın geçmişte doğrulanan etiket oranı), hatalı etiket patlamasını doğru ama beklenmedik etiketlerden (drift) ayırabilir. Pekiştirme ve oy ağırlığı bu güvene bağlanırsa, hem pekiştirme sonrası hasar hem drift korunur. | B1 ve D1'in ortak gerilimi. | Patlama hasarı +600'de < 0,5 puan, drift ve karışık doğruluk Stage 06 seviyesinde. |

