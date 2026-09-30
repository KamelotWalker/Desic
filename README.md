# Desic

**Kendi kendine öğrenen, açıklanabilir karar motoru** — gerçek zamanlı geri bildirim dashboard'u, kendi veri setinizle eğitim ve kendi AI API anahtarınızla veri üretimi.

Desic klasik kural motorlarının (Drools/JESS tarzı "EĞER … İSE …" kuralları) şeffaflığını, sürekli öğrenen bir karar ağacıyla birleştirir:

- Her karar **neden** verildiğini gösterir (ağaçtaki yol: `credit_score > 1423 → employment ≠ unemployed → approve`).
- Kullanıcı "bu doğru / yanlış, doğrusu X" dediği anda model **tek örnekten** öğrenir. Yeniden eğitim yok, batch yok.
- Doğru cevap zamanla değişirse (**concept drift**) Desic bunu fark eder ve kendini günceller.
- Öğrenilen yollar okunabilir kurallara dönüşür; istediğinizi tek tıkla **sabit kurala** çevirip modeli ezebilirsiniz.

```
            ┌────────────── Dashboard (gerçek zamanlı, WebSocket) ──────────────┐
            │ karar ver · geri bildirim · doğruluk grafiği · ağaç · kurallar     │
            └──────────────┬─────────────────────────────────────▲─────────────┘
 senin servisin ──HTTP──►  │  /decide  →  sabit kurallar ──eşleşme yok──► öğrenen model
                           │  /feedback →  prequential ölçüm → model.learn_one() │
                           │  /datasets →  CSV/JSON yükle → akış halinde eğit    │
                           │  /generate →  LLM (senin anahtarın) → sentetik veri │
                           └──────────────── SQLite (modeller, karar günlüğü) ───┘
```

## Hızlı başlangıç

```bash
pip install -e .            # Python 3.10+
desic demo                  # örnek kredi-onay modeli + veri seti oluşturur
desic serve                 # http://127.0.0.1:8000
```

Dashboard'da **Models → loan_demo** açın. Başka bir terminalde gerçek kullanıcıları simüle edin ve grafiğin canlı değişmesini izleyin:

```bash
python examples/feedback_simulator.py --n 1000 --drift 500   # 500. karardan sonra "banka politikası" değişir
```

Docker ile:

```bash
docker build -t desic . && docker run -p 8000:8000 -v desic-data:/data desic
```

## Öğrenme motoru nasıl çalışıyor?

Tamamı saf Python, ek ML kütüphanesi yok (`desic/core`).

| Parça | Ne yapar |
|---|---|
| **Hoeffding ağacı** (`tree.py`) | VFDT (Domingos & Hulten, 2000). Her örnek bir kez görülür ve atılır; bir yaprak ancak Hoeffding sınırı "en iyi bölme gerçekten en iyisi" dediğinde bölünür. Sayısal özellikler için sınıf başına Gauss tahmincisi, kategorikler için sayaçlar. Yapraklarda *Naive Bayes Adaptive* tahmin. Eksik değerler desteklenir. |
| **Drift algılama** (`drift.py`) | DDM (Gama, 2004). Hata oranı yükselince önce *uyarı* verip arka planda yeni bir ağaç eğitmeye başlar, *drift* kesinleşince onu devreye alır. |
| **Adaptive Random Forest** (`model.py`) | İsteğe bağlı: Poisson(6) yeniden örnekleme + rastgele alt özellik kümeleri ile N ağaç, her birinin kendi drift dedektörü; oylar son doğruluğa göre ağırlıklı. Daha isabetli, açıklama en isabetli ağaçtan gelir. |
| **Prequential ölçüm** | Her etiket öğrenilmeden *önce* puanlanır, yani dashboard'daki doğruluk hep "görülmemiş veri" doğruluğudur. Geri bildirimde, kullanıcıya gösterilmiş olan tahmin puanlanır. |
| **Sabit kurallar** (`rules.py`) | Öncelik sıralı, `==, !=, >, >=, <, <=, in, not_in, contains, is_missing` operatörleri. Kurala uyan kararlarda da model arka planda öğrenmeye devam eder; dashboard her kuralın kaç kez insan tarafından **ezildiğini** gösterir. |
| **Aktif öğrenme** | "Review queue" etiketsiz kararları modelin en emin olmadığı sırayla listeler; en çok bilgi kazandıran geri bildirimler önce. |

## Veri setleri

- **Kendi verin:** CSV, TSV, `;` ayraçlı CSV, JSON dizisi veya JSON Lines (≤ 50 MB). Kolon tipleri otomatik çıkarılır; hedef (karar) kolonunu seçip tek tıkla eğitirsin. Hold-out oranı ve kaç tur geçileceği ayarlanabilir; mevcut bir modeli yeni veriyle eğitmeye devam etmek de mümkün.
- **AI ile üret (kendi API anahtarınla):** Kararı düz metinle anlat ("KOBİ kredisi onayla/incele/reddet") → LLM özellikleri, sınıfları ve karar mantığını tasarlar → tasarımı düzenle → satırlar 40'lık gruplar halinde, şemaya göre doğrulanarak üretilir → istersen hemen model eğitilir.
  - **Anthropic (Claude)** resmi SDK ile, yapılandırılmış çıktı (JSON schema) kullanılarak. Varsayılan model `claude-opus-5-5`.
  - **OpenAI-uyumlu her uç nokta**: OpenAI, Google Gemini, Groq, OpenRouter, yerel Ollama / LM Studio …
  - Anahtar yalnızca o istek için sağlayıcıya iletilir; Desic **saklamaz, loglamaz**. Boş bırakılırsa sunucunun `ANTHROPIC_API_KEY` ortam değişkeni kullanılır. "Bu tarayıcıda hatırla" işaretlenirse yalnızca tarayıcının `localStorage`'ında durur.

## API

Etkileşimli referans: `http://127.0.0.1:8000/docs`. Dashboard'daki **Integrate** sekmesi seçili model için hazır `curl`/Python örnekleri üretir.

```bash
# 1) karar iste
curl -X POST localhost:8000/api/models/loan_demo/decide -H 'Content-Type: application/json' \
  -d '{"features": {"monthly_income": 45000, "loan_amount": 100000, "credit_score": 1700,
                    "employment": "salaried", "debt_ratio": 0.2, "city": "izmir"}}'
# → {"id": "…", "prediction": "approve", "confidence": 0.92, "explanation": {"path": [...]}, ...}

# 2) gerçek sonucu bildir → model anında öğrenir, dashboard anında güncellenir
curl -X POST localhost:8000/api/models/loan_demo/feedback -H 'Content-Type: application/json' \
  -d '{"decision_id": "…", "label": "approve"}'
```

| Uç nokta | Açıklama |
|---|---|
| `GET/POST /api/models` | modelleri listele / şemayla yeni model oluştur (`kind`: `tree` \| `forest`) |
| `GET/DELETE /api/models/{name}` · `POST …/reset` | detay (metrikler, geçmiş, önem, olaylar) / sil / öğrendiklerini sıfırla |
| `POST …/decide` · `POST …/feedback` · `POST …/learn` | karar · geri bildirim · etiketli satırlarla toplu öğrenme |
| `GET …/decisions?pending=true&uncertain_first=true` | karar günlüğü / aktif öğrenme kuyruğu |
| `GET …/tree` · `GET …/learned-rules` · `PUT …/rules` | ağaç · çıkarılmış kurallar · sabit kurallar |
| `POST /api/datasets` · `GET /api/datasets/{id}` · `POST …/{id}/train` | yükle · profil+önizleme · eğitim işi başlat |
| `POST /api/generate/design` · `POST /api/generate/dataset` | LLM ile şema tasarla · veri üret (+ opsiyonel eğitim) |
| `GET /api/jobs/{id}` · `WS /ws` | arka plan işleri · gerçek zamanlı olay akışı |

Python içinde doğrudan da kullanılabilir:

```python
from desic.service import Desic
from desic.core import Schema, Feature

desic = Desic("./data")
desic.create_model("churn", Schema([Feature("tenure"), Feature("plan", "categorical")], target="churn"))
d = desic.decide("churn", {"tenure": 3, "plan": "basic"})
desic.feedback("churn", d["id"], "yes")
```

## Geliştirme

```bash
pip install -e '.[dev]'
pytest
desic serve --reload
```

```
desic/
  core/       stats.py · tree.py · drift.py · model.py · rules.py · schema.py   ← öğrenme motoru
  service.py  modeller, kararlar, geri bildirim, işler, olay yayını
  storage.py  SQLite (modeller pickle olarak, karar günlüğü, veri setleri)
  generation.py  LLM ile şema tasarımı ve veri üretimi
  api.py      FastAPI + WebSocket
  static/     dashboard (bağımlılıksız HTML/CSS/JS)
```

## Yol haritası

- Regresyon (sayısal karar) için Hoeffding regresyon ağacı
- Kullanıcı/rol yönetimi ve API anahtarları (şu an yerel/güvenilir ağ için tasarlandı)
- LLM'i "öğretmen" olarak kullanma: düşük güvenli kararları AI'ın etiketlemesi, insanın onaylaması
- Sıcak yolu (ağaç içinden geçiş, istatistik güncelleme) C++ / pybind11 ile hızlandırma
- Model sürümleme ve A/B karşılaştırma

> ⚠️ Desic şu an kimlik doğrulaması olmadan çalışır; `desic serve` varsayılan olarak yalnızca `127.0.0.1`'i dinler. İnternete açmadan önce önüne bir kimlik doğrulama katmanı koyun. Model durumları `pickle` ile saklanır; yalnızca kendi oluşturduğunuz veri klasörlerini yükleyin.

---

### English summary

Desic is a self-learning, explainable decision engine: an incremental Hoeffding tree (or adaptive random forest) with DDM drift detection learns from every piece of user feedback in real time, while priority-ordered hard rules give rule-engine style control. It ships with a realtime dashboard (live decision feed, prequential accuracy chart, tree/rule views, active-learning review queue), dataset upload/training, and synthetic dataset generation using your own Anthropic or OpenAI-compatible API key (keys are never stored). `pip install -e . && desic demo && desic serve`.

MIT License.
