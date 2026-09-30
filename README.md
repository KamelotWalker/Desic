# Desic

**Kendi kendine öğrenen, kalibre typed-decision motoru (System One).**
Jev ve Laya gibi Desic de serbest metin üretmez: bir `state` (metin veya JSON) ve tipli sorular alır, izin verilen cevaplar üzerinde **olasılık dağılımı** döndürür. Farkı: her kullanıcı geri bildiriminden **anında öğrenir**, emin olmadığında **çekimser kalır** ve gerekirse kendi LLM anahtarınızla çalışan bir "öğretmene" (System 2) danışıp ondan da öğrenir.

```
                     ┌────────────── Dashboard (gerçek zamanlı, WebSocket) ───────────────┐
                     │ playground · kalibrasyon · review queue · feedback log · sürümler   │
                     └───────────────┬─────────────────────────────────────▲──────────────┘
 servisin ─ POST /v1/decide ────────►│ state + { soru: choice | score | noul }            │
                                     ▼                                                    │
                        sabit kurallar ──eşleşme yok──► ÖĞRENCİ (System 1)                │
                                                        prior · linear · tree · memory   │
                                                        (+ neural: Laya tarzı encoder)   │
                                                        → Hedge karışımı → sıcaklık kalib.│
                                                        → tanıdıklık → güven / abstain    │
                                        abstain ise ──► ÖĞRETMEN (System 2): Claude,      │
                                                        OpenAI-uyumlu LLM ya da           │
                                                        Jev-uyumlu API (ör. Laya)         │
                                                        cevabı sunulur + öğrenciye öğretilir
 servisin ─ POST /v1/feedback ──────► değiştirilemez feedback log ─► öğrenci anında günceller
                                      (retract · rebuild · snapshot · rollback)
```

## Hızlı başlangıç

```bash
pip install -e .            # Python 3.10+
desic demo                  # destek talepleri (metin) + kredi başvuruları (JSON) için 3 soru
desic serve                 # http://127.0.0.1:8000
```

**Playground**'da "Support ticket" örneğiyle **Decide**'a bas. Sonra başka bir terminalde gerçek kullanıcıları simüle et; soru sayfasındaki öğrenme eğrisi ve güvenilirlik diyagramı canlı değişir:

```bash
python examples/feedback_simulator.py --n 1000 --drift 500   # 500. karardan sonra kredi politikası değişir
```

Öğretmen (isteğe bağlı): dashboard'da **Teacher** sayfasından sağlayıcı ve anahtar gir, ya da:

```bash
ANTHROPIC_API_KEY=sk-ant-... desic serve                       # Claude (varsayılan: claude-opus-5-5)
DESIC_TEACHER_PROVIDER=openai DESIC_TEACHER_BASE_URL=https://api.openai.com/v1 \
DESIC_TEACHER_MODEL=<model> DESIC_TEACHER_API_KEY=... desic serve
DESIC_TEACHER_PROVIDER=jev_compatible DESIC_TEACHER_BASE_URL=http://localhost:9000/v1/decide desic serve  # ör. self-host Laya
```

Nöral öğrenci (isteğe bağlı): `pip install -e '.[neural]'` → dashboard'da **Neural** sayfası.

### Bilgisayarsız: GitHub Codespaces (telefondan da çalışır)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/KamelotWalker/Desic)

Repo sayfasında **Code → Codespaces → Create codespace** (ya da yukarıdaki rozet). Kurulum, demo verisi ve sunucu `.devcontainer/devcontainer.json` ile otomatik başlar; 8000 portu **private** olarak yönlendirilir, yani adres yalnızca senin GitHub hesabınla açılır. Adres **Ports** sekmesinde görünür (`https://<codespace-adı>-8000.app.github.dev`). Kullanmadığında codespace'i durdur (ücretsiz kotadan yer).

Docker: `docker build -t desic . && docker run -p 8000:8000 -v desic-data:/data desic`

## API (Jev istek biçimi)

```bash
curl -s -X POST localhost:8000/v1/decide -H 'Content-Type: application/json' -d '{
  "state": "Kartımdan iki kez ödeme çekildi, acil iade istiyorum",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "payments, refunds", "technical": "bugs", "sales": "pricing"}},
    "severity":   {"type": "score",  "criteria": {"low": "cosmetic", "medium": "workaround exists", "high": "blocking"}},
    "urgent":     {"type": "noul",   "instructions": "The customer needs a response today."}
  },
  "explain": true
}'
```

```jsonc
{
  "id": "dec_…",
  "answers": {
    "department": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.97, "sales": 0.02, "technical": 0.01},
                   "confidence": 0.97, "abstain": false, "source": "student", "explanation": {…}},
    "severity":   {"type": "score", "score": 0.4, "level": "low", "probabilities": {…}, …},
    "urgent":     {"type": "noul", "probability": 0.95, "answer": true, …}
  }
}
```

- Yeni bir soru adı ilk kullanımda otomatik kaydedilir; bilinen soru için `{}` göndermek yeterli.
- `choice` sorusuna feedback ile **yeni bir cevap** öğretilebilir (open world); `score` seviyeleri sabittir.
- `escalate`: `auto` (sorunun öğretmen ayarı) · `never` · `always`. `abstain_threshold` istek bazında değiştirilebilir.

Gerçek sonuç belli olunca:

```bash
curl -s -X POST localhost:8000/v1/feedback -H 'Content-Type: application/json' \
  -d '{"decision_id": "dec_…", "answers": {"department": "billing", "urgent": true, "severity": 2}}'
```

| Uç nokta | Açıklama |
|---|---|
| `POST /v1/decide` · `POST /v1/feedback` · `GET /v1/decisions/{id}` | karar · geri bildirim · karar kaydı |
| `GET/POST /v1/questions` · `GET/PATCH/DELETE /v1/questions/{name}` | sorular; ayarlar: `abstain_threshold`, `teacher_mode` (`off`/`on_abstain`/`always`), `teacher_weight` |
| `POST …/{name}/learn` | etiketli örneklerle toplu öğrenme `{"examples": [{"state": …, "answer": …}]}` |
| `GET …/{name}/decisions?pending=true&uncertain_first=true` | canlı akış / aktif öğrenme kuyruğu |
| `PUT …/{name}/rules` | sabit kurallar (JSON yolu, ör. `applicant.credit_score`, ya da `$text`) |
| `GET …/{name}/feedback` · `POST /v1/feedback/{id}/retract` · `POST …/{name}/rebuild` | değiştirilemez log · hatalı etiketi geri çek · logdan yeniden kur |
| `GET/POST …/{name}/snapshots` · `POST …/{name}/rollback` | sürümler ve geri alma |
| `GET/PUT/DELETE /v1/teacher` · `POST /v1/teacher/test` | öğretmen ayarı (anahtar sadece bellekte) |
| `POST /v1/datasets` · `POST /v1/datasets/{id}/train` · `POST /v1/datasets/{id}/distill` | yükle · eğit · öğretmene etiketlet |
| `POST /v1/generate/design` · `POST /v1/generate/examples` | LLM ile soru tasarla · örnek üret (+ eğit) |
| `GET /v1/neural` · `PATCH /v1/neural/config` · `POST /v1/neural/train` | nöral öğrenci durumu · ayarlar · aday eğitimi |
| `POST /v1/neural/checkpoints/{id}/{promote\|reject\|retire\|delete}` | checkpoint yaşam döngüsü |
| `GET /v1/jobs/{id}` · `WS /ws` | arka plan işleri · canlı olaylar |

Etkileşimli referans: `http://127.0.0.1:8000/docs`. Python içinden doğrudan: `desic.service.Desic`.

## Öğrenci nasıl çalışıyor? (`desic/core`)

Saf Python, ek ML bağımlılığı yok. Her soru için ayrı bir öğrenci:

| Parça | Ne yapar |
|---|---|
| **Özellikler** (`features.py`) | Metin: kelime, 5 harflik kök (Türkçe gibi eklemeli diller için hafif stemmer), bigram; büyük/küçük ve ASCII katlama (`şikayet` = `sikayet`). JSON: `customer.plan=pro` gibi alanlar, standartlaştırılmış sayılar. |
| **linear** uzmanı | Çevrimiçi softmax regresyon, log loss + AdaGrad (öğrenme hızı tabanı sayesinde drift'e uyum). Öğretmenden gelen *yumuşak* olasılıkları da öğrenir. |
| **tree** uzmanı | JSON alanları üzerinde Hoeffding ağacı (VFDT) + DDM drift dedektörü; `credit_score > 1445` gibi eşikler. |
| **memory** uzmanı | En yakın etiketli örnekler (ters indeksli kosinüs). **Sıcak katman**: tek bir düzeltme, aynı/benzer durum için bir sonraki cevabı hemen değiştirir. |
| **prior** uzmanı | Temel oranlar. |
| **Karışım** (`experts.py`) | Log loss üzerinde Hedge (η=1 → Bayesçi karışım), *uyuyan uzmanlar* ve *fixed share* ile drift'ten sonra hızlı toparlanma. Log loss kesin uygun (strictly proper) bir skorlama kuralı: uzmanlar dürüst olasılık için ödüllenir. |
| **Kalibrasyon** (`calibration.py`) | Son 500 etiket üzerinde çevrimiçi sıcaklık ölçekleme; T ∈ [0.5, 4] ile sınırlı (tanıdık veride öğrenilen sıcaklık, hiç görülmemiş girdide aşırı güven üretmesin). |
| **Tanıdıklık** | State'teki kanıtın ne kadarı eğitimde görüldü? %50'nin altındaysa dağılım "bilmiyorum"a çekilir → çekimser kalır → öğretmene gider. |
| **Ölçüm** | Prequential: her insan etiketi, kullanıcıya *gösterilmiş* olasılıklarla öğrenilmeden önce puanlanır. Accuracy, NLL, Brier, ECE, RPS (score), güvenilirlik diyagramı, risk–coverage, abstain ve öğretmen oranı. Öğretmen etiketleri metriğe girmez. |

## Nöral öğrenci (Laya tarzı, `desic/neural`)

Çevrimiçi uzmanlar **hızlı katman**: her geri bildirimden anında öğrenir. Nöral öğrenci **yavaş katman**: feedback log'dan periyodik olarak eğitilen, Laya mimarisinde bir encoder + decision head.

```
[CLS] choice: Which team? [MASK] billing: payments [MASK] technical: bugs [MASK] sales: pricing [SEP] <state> [SEP]
        │ encoder (ModernBERT-large / mmBERT / herhangi bir HF encoder / yerleşik küçük encoder)
        │ 2 katmanlı decision transformer
        ├─ her [MASK] üzerinde option scorer → softmax → bu sorunun cevapları üzerinde dağılım
        └─ [CLS] üzerinde act/escalate head → P(en üstteki cevap doğru)
```

- **Veri:** geri çekilmemiş tüm etiketler, (soru, state) başına bir tane; insan > dataset > öğretmen. Öğretmen olasılıkları yumuşak hedef (distillation).
- **Kayıp:** log score (strictly proper) + sıralı sorular için RPS + act head; isteğe bağlı **RLCD tarzı** aşama (logitlere Gauss gürültüsü, grup örnekleme, log-score ödülü, grup-ortalama baseline ile REINFORCE). Encoder ve head için ayrı öğrenme hızları.
- **Kalibrasyon:** doğrulama diliminde soru başına sıcaklık.
- **Kapılar:** (1) *offline* — hiçbir checkpoint'in eğitilmediği, hash ile sabit test diliminde taban oranları yenmeli ve aktif checkpoint'e kaybetmemeli; (2) *gölge* — sonraki N etiketli kararda sunulmadan tahmin eder, log loss'u kullanıcılara sunulana yakınsa terfi eder. (3) *aktif* — her sorunun karışımına `neural` uzmanı olarak girer; Hedge ne kadar güvenileceğine soru bazında karar verir. İstendiğinde emekliye ayrılır.
- **Backbone:** `scratch` (indirme yok, CPU'da saniyeler), `answerdotai/ModernBERT-large` (Laya İngilizce), `jhu-clsp/mmBERT-base` (Türkçe dahil 100+ dil), ya da herhangi bir HF encoder id / yerel yol. Önceden eğitilmiş encoder'lar ilk kullanımda Hugging Face'ten indirilir ve GPU ister.
- **Güvenlik notu:** yerleşik (önceden eğitilmemiş) encoder, anlamsız bir metne bile %99 güven verebilir — nöral ağların klasik dağılım-dışı aşırı güveni. Bu yüzden "tanıdıklık" koruması yalnızca **önceden eğitilmiş** bir encoder aktifken gevşer.

### Jev / Laya ile karşılaştırma

| | Jev | Laya | Desic |
|---|---|---|---|
| Arayüz | state + choice/score/noul | aynı | aynı (Jev istek biçimi) |
| Model | kapalı, hosted | ModernBERT + decision head (421M) | çevrimiçi uzman karışımı + isteğe bağlı Laya tarzı nöral öğrenci |
| Sıfırdan (zero-shot) bilgi | güçlü | orta | yok: öğretmen + veri ile öğrenir |
| Kullanıcı geri bildiriminden öğrenme | hayır (müşteri fine-tune yok) | offline fine-tune | **her geri bildirimde, anında** |
| Kalibrasyon | RLCD | proper scoring + sıcaklık | log-loss Hedge + çevrimiçi sıcaklık + tanıdıklık |
| Çekimser kalma / yükseltme | eşik kodda | act/escalate head | birinci sınıf `abstain` + otomatik öğretmen |
| Güvenlik | — | — | değiştirilemez log, retract, rebuild, snapshot/rollback |

Desic, Laya'yı (Jev-uyumlu bir sunucu arkasında) **öğretmen** olarak kullanabilir: Laya'nın zero-shot bilgisi + Desic'in anlık öğrenmesi.

## Benchmark: Banking77

77 bankacılık niyeti, 10.003 eğitim / 3.080 test (`python examples/benchmark_banking77.py [--neural] [--sorted]`, CPU, GPU yok):

| Kurulum | Doğruluk | Log loss | ECE |
|---|---|---|---|
| Çevrimiçi öğrenci, sınıf başına 10 örnek | %57,2 | 1,70 | 0,122 |
| Çevrimiçi öğrenci, sınıf başına 50 örnek | %82,6 | 0,62 | 0,011 |
| Çevrimiçi öğrenci, tüm veri (94 sn) | **%88,5** | 0,43 | **0,013** |
| Yerleşik nöral öğrenci tek başına (6 epoch, 13,5 dk) | %68,4 | 1,15 | 0,015 |
| Çevrimiçi + nöral karışım | %88,3 | 0,44 | 0,035 |
| Çevrimiçi öğrenci, **sınıfa göre sıralı akış** (stres testi) | %15,8 | 3,37 | 0,474 |

Seçici tahmin (çevrimiçi öğrenci, tüm veri): yalnızca ≥%70 eminken cevap verirse sorunun %85,7'sine %95,1, ≥%90 eminken %73,3'üne %97,8 doğrulukla cevap veriyor.

### Akış senaryoları (`desic eval`)

Doğruluk tek başına yetmiyor: model akış boyunca nasıl öğreniyor, hatalı etiketten ve drift'ten nasıl etkileniyor? Değerlendirme düzeneği bunu 7 senaryo × 3 tohumla ölçüyor ([aşama 02](benchmarks/stage-02-eval-harness/)):

| Senaryo | Sonuç |
|---|---|
| %1 / %5 hatalı etiket | Doğruluk %88,5 / %87,9 (dayanıklı); %5'te ECE 0,085 |
| 30 hatalı etiketlik patlama | Kurban sınıf %77 → %1, diğer sınıflarda da 3–5 puan hasar; geri alma = tüm kaydı (5.000 olay) yeniden oynatmak |
| 10 sınıfın anlamı değişiyor (drift) | Yarı toparlanma 3.000 etiket; drift dedektörü tetiklenmiyor |
| Öğretmenli soğuk başlangıç | Öğretmene bağımlılık %95 → %46 |

Tüm aşamaların ölçümleri, ham loglar ve JSON sonuçları: [`benchmarks/`](benchmarks/).

Referans: yayınlanmış sonuçlarda (Casanueva ve ark., 2020) tam veriyle ince ayarlı BERT / ConveRT yaklaşık %93. Desic'in önceden eğitilmiş bir dil bilgisi yok; bu fark beklenen bir fark. Önceden eğitilmiş bir backbone (mmBERT / ModernBERT) ile nöral öğrenci GPU'da ayrıca ölçülmeli. Sıralı akıştaki çöküş bilinen bir zayıflık (catastrophic forgetting) ve bir sonraki geliştirme hattının ilk hedefi.

## Veri

- **Kendi verin:** CSV/TSV/JSON/JSONL (≤ 50 MB). Cevap kolonunu ve state kolonlarını seç; state metin (kolonlar birleştirilir) ya da JSON nesnesi olur. Hold-out ile kalibrasyon dahil değerlendirme.
- **Distill:** etiketsiz satırları öğretmen etiketler, öğrenci onun olasılıklarından öğrenir (System 2 → System 1).
- **Üret:** kararı düz metinle anlat → LLM soruyu tasarlar (tip, cevaplar, state biçimi, uzman kuralları) → düzenle → örnekler 20'lik gruplar halinde yazılır, şemaya göre doğrulanır → istersen hemen eğitilir.
- Anahtarlar yalnızca sunucu belleğinde (ve istersen tarayıcında) tutulur; veritabanına ve loglara yazılmaz.

## Geliştirme

```bash
pip install -e '.[dev]'
pytest                      # 62 test: çekirdek, API, öğretmen/üretim, nöral öğrenci (torch yoksa atlanır), değerlendirme
desic eval --seeds 1        # akış senaryoları (Banking77)
desic serve --reload
```

```
desic/
  core/        features · experts · calibration · task · tree · model (adaptif ağaç) · drift · rules
  eval/        data (Banking77) · metrics (prequential, seçici risk, forgetting, half-life) · scenarios · run
  neural/      text (girdi biçimi) · model (encoder + decision head) · train (kayıplar, RLCD, kalibrasyon) · runtime (kapılar)
  service.py   kararlar, feedback, öğretmen, işler, snapshot/rebuild
  llm.py       öğretmen sağlayıcıları (Anthropic SDK, OpenAI-uyumlu, Jev-uyumlu)
  generation.py  LLM ile soru tasarımı ve örnek üretimi
  storage.py   SQLite: öğrenciler, snapshot'lar, kararlar, feedback log, veri setleri
  api.py       FastAPI + WebSocket
  static/      dashboard (bağımlılıksız HTML/CSS/JS)
```

## Yol haritası

- Nöral öğrencide çok dilli yönlendirici (Laya Router gibi dile göre checkpoint seçimi) ve ONNX/quantization ile hızlı çıkarım.
- Gölge testinde eşleştirilmiş istatistiksel test (şu an ortalama log loss + marj).
- Multi-label ve sıralama (rank) primitive'leri, hiyerarşik choice (255+ seçenek).
- Sıcak yolun (özellik çıkarma, uzman skorlama) C++/pybind11 ile hızlandırılması.
- Kimlik doğrulama ve API anahtarları (şu an yerel/güvenilir ağ için; `desic serve` varsayılan olarak yalnızca `127.0.0.1`'i dinler).

> Öğrenci durumları `pickle` ile saklanır; yalnızca kendi oluşturduğunuz veri klasörlerini yükleyin.

---

### English summary

Desic is a self-learning **typed-decision (System One) engine**: like Jev and Laya it takes a state plus typed questions (`choice`, `score`, `noul`) and returns calibrated probability distributions, never free text. Unlike them it learns from **every piece of feedback in real time** (a mixture of online experts — linear, Hoeffding tree, nearest-neighbour memory, prior — combined by log-loss Hedge with fixed share, then temperature-calibrated), optionally joined by a **Laya-style neural student** (encoder + option-marker decision head, trained from the feedback log with proper scoring / RLCD, gated offline and in shadow before promotion), **abstains** when unsure or when the input is unfamiliar, and can escalate to a **teacher** (Claude, any OpenAI-compatible LLM, or a Jev-compatible API such as a self-hosted Laya) whose answers are distilled back into the student. Feedback is an append-only log with retract / rebuild / snapshot / rollback. The dashboard shows ECE, Brier, NLL, reliability diagrams and risk–coverage curves. `pip install -e . && desic demo && desic serve`.

MIT License.
