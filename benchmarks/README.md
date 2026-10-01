# Desic benchmarks

Her geliştirme aşamasının ölçümleri burada **ayrı bir klasörde** ve değiştirilmeden saklanır. Böylece aşamalar birbiriyle karşılaştırılabilir. Yeni bir aşama eski sonuçların üzerine yazmaz, yeni bir klasör açar.

| Aşama | Klasör | Ne ölçüldü | Özet |
|---|---|---|---|
| 00 | [`stage-00-baseline-v0.2`](stage-00-baseline-v0.2/) | v0.2 çevrimiçi öğrenci + yerleşik nöral öğrenci, Banking77 | %88,5 doğruluk, ECE 0,013; sıralı akışta %15,8'e çöküş |
| 01a | [`stage-01a-adagrad-g0-rejected`](stage-01a-adagrad-g0-rejected/) | İlk kullanıcı denemesindeki aşırı güven hatası için AdaGrad `g0` denemesi | **Reddedildi:** az veride ECE 0,122 → 0,363, sıralı akış %15,8 → %1,5 |
| 01b | [`stage-01b-evidence-scaled-updates`](stage-01b-evidence-scaled-updates/) | Adımın etiketin kanıt değeriyle ölçeklenmesi + 3 düzeltme | Hata giderildi (%55'lik etiket: 0,94 → 0,35), Banking77'de regresyon yok |
| 02 | [`stage-02-eval-harness`](stage-02-eval-harness/) | Faz 0: değerlendirme düzeneği, 7 akış senaryosu × 3 tohum | Karışık %88,8 · ECE 0,015; 30 hatalı etiket bir sınıfı ele geçiriyor (%77 → %1), geri alma = 5.000 olay; drift yarı ömrü 3.000 etiket; forgetting index 0,63 |
| 03 | [`stage-03-reversible-patches`](stage-03-reversible-patches/) | Faz 2: geri alınabilir patch katmanı (`desic+patch`) ve temel model, aynı akışlar | Sıralı %13,7 → %80,2; patlama hasarı diğer sınıflarda 3,8 → 0,5 puan; geri alma 29,6 sn → 0,1 sn; karışık %88,9 (regresyon yok). Kötüleşen: drift (3.000 → 3.417), yalan tekrarlama (%7 → %23) |
| 04 | [`stage-04-service-integration`](stage-04-service-integration/) | Patch katmanı dashboard/API'de; deneme süresi etiketlerin %10'u (20–500); kontrol: temel model + 10× hafıza | Karışık %89,3 ve biriken hata artık temel modelden iyi; sıralı %80,2 (10× hafızalı temel model: %12,7); hâlâ kötü: drift (3.500), yanlış tekrarlama (~%23), öğretmenli senaryoda cevap doğruluğu (%85,6) |
| 05 | [`stage-05-trust-gate-variants-rejected`](stage-05-trust-gate-variants-rejected/) | Kalan üç zayıflık için 4 güven/kapı varyantı (tek tohum) | **Reddedildi:** hiçbiri drift, yanlış tekrarlama ya da öğretmenli doğrulukta belirgin iyileşme sağlamadı; sebep tahmini yanlıştı |

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

### Akış metrikleri (aşama 02'den itibaren, `desic eval`)

| Metrik | Anlamı | İyi yön |
|---|---|---|
| Kümülatif log loss | Akıştaki her etiket, öğrenilmeden *önce* kullanıcıya sunulan olasılıkla puanlanır (prequential); tüm akış boyunca ortalama. Hem ne kadar hızlı öğrendiğini hem ne kadar dürüst emin olduğunu ölçer | ↓ |
| Risk@%80 | Model en emin olduğu %80'lik kısma cevap verirse hata oranı (seçici risk) | ↓ |
| AURC | Risk–kapsam eğrisinin altındaki alan; eşikten bağımsız seçici risk | ↓ |
| Kapsam / cevaplanan doğruluk | Varsayılan çekimserlik kuralıyla (güven < 0,6 veya tanımadığı girdi) cevap verdiği oran ve o cevapların doğruluğu | ↑ |
| Forgetting index | Sıralı akışta her sınıfın gördüğü en iyi test doğruluğu ile final doğruluğu arasındaki fark, ortalama (Chaudhry ve ark., 2018) | ↓ |
| Adaptation half-life | Bir şoktan (drift veya hatalı etiket patlaması) sonra kaybedilen başarımın yarısını geri kazanmak için gereken etiket sayısı | ↓ |
| Teacher dependency | Öğretmene giden kararların oranı; zamanla düşmesi gerekir | ↓ |
| Undo maliyeti | Hatalı etiketleri geri almak için yeniden oynatılması gereken olay sayısı ve süre | ↓ |

### Senaryolar

| Senaryo | Ne yapıyor |
|---|---|
| `shuffled` | Karışık akış; öğrenme eğrisi (500 / 1000 / 2500 / 5000 / tümü) |
| `sorted` | Sınıflar sırayla gelir (11 grup); her gruptan sonra test, forgetting index |
| `noise-1%`, `noise-5%` | Etiketlerin %1 / %5'i sessizce yanlış; model yalan söylendiği mesajlarda yalanı tekrarlıyor mu (`poison_repeated`) |
| `burst` | Akışın ortasında A sınıfının 30 mesajı art arda B olarak etiketlenir; hasar, temiz etiketlerle toparlanma ve bugünkü geri alma maliyeti |
| `drift` | Akışın ortasında 10 sınıfın anlamı döner (c₁→c₂→…→c₁₀→c₁); yeni anlamı öğrenme hızı ve diğer sınıfların kararlılığı |
| `teacher` | Soğuk başlangıç, hiç etiket yok; çekimser kalınan mesajları simüle öğretmen (kalibre, ~%80 doğru, yumuşak etiket) etiketler, insan %5'ini kontrol eder |

Her senaryo 3 tohumla koşulur (akış sırası, gürültü, saldırılan sınıf ve öğretmen hataları tohumdan türetilir) ve ortalama ± standart sapma raporlanır. Her öğrenci varyantı birebir aynı akışı görür.

```bash
desic eval --out sonuc.json                         # tüm senaryolar, 3 tohum (4 çekirdekte ~10 dk)
desic eval --scenarios burst,drift --seeds 1        # bir kısmı
desic eval --limit 2000 --seeds 1                   # hızlı deneme
```
