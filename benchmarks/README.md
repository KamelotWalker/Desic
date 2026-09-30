# Desic benchmarks

Her geliştirme aşamasının ölçümleri burada **ayrı bir klasörde** ve değiştirilmeden saklanır. Böylece aşamalar birbiriyle karşılaştırılabilir. Yeni bir aşama eski sonuçların üzerine yazmaz, yeni bir klasör açar.

| Aşama | Klasör | Ne ölçüldü | Özet |
|---|---|---|---|
| 00 | [`stage-00-baseline-v0.2`](stage-00-baseline-v0.2/) | v0.2 çevrimiçi öğrenci + yerleşik nöral öğrenci, Banking77 | %88,5 doğruluk, ECE 0,013; sıralı akışta %15,8'e çöküş |

## Her aşama klasöründe

- `README.md` — sonuç tablosu, bulgular, kullanılan commit, ortam ve tekrar üretme komutları
- `*.json` — script'in yazdığı ham sonuçlar ve çalıştırma bilgisi (tarih, commit, Python/torch sürümü, CPU/GPU, argümanlar)
- `logs/*.txt` — script'in birebir konsol çıktısı

## Tekrar üretme

```bash
pip install -e .                    # nöral kısım için: pip install -e '.[neural]'
python examples/benchmark_banking77.py --out sonuc.json            # karışık akış, çevrimiçi öğrenci
python examples/benchmark_banking77.py --sorted --out sonuc.json   # sınıfa göre sıralı akış (stres testi)
python examples/benchmark_banking77.py --neural --out sonuc.json   # + yerleşik nöral öğrenci (CPU'da ~15 dk)
```

Veri: [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI), script'in her çalışmada GitHub'dan indirdiği 10.003 eğitim / 3.080 test örneği, 77 niyet. Test kümesi hiçbir zaman eğitimde kullanılmaz. Karıştırma sabit bir tohumla yapılır (`random.Random(0)`), bu yüzden çevrimiçi sonuçlar deterministiktir.

## Metrikler

| Metrik | Anlamı | İyi yön |
|---|---|---|
| Doğruluk | En olası cevabın doğru olma oranı | ↑ |
| Log loss (NLL) | Doğru cevaba verilen olasılığın negatif logaritması; "emin ama yanlış" cevabı ağır cezalandırır | ↓ |
| ECE | Beklenen kalibrasyon hatası: "%80 eminim" dediğinde gerçekten %80 doğru mu? | ↓ |
| Kapsam @ eşik | Model yalnızca bu güvenin üstündeyken cevap verirse, soruların ne kadarını cevaplar | ↑ |
| Cevaplanan doğruluk @ eşik | O cevapların doğruluğu (seçici risk = 1 − bu değer) | ↑ |

Sonraki aşamalarda eklenecekler: kümülatif log loss, adaptation half-life (drift sonrası kaybedilen performansın yarısını kaç geri bildirimde geri kazandığı), forgetting index, hatalı etiket enjeksiyonu altında toparlanma, teacher dependency.
