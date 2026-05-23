# Song Miner (Local)

Bu araç, lokal bilgisayarda şunları yapar:
- Şarkıyı YouTube üzerinden indirir (WAV)
- Sinyal özelliklerini çıkarır (tempo, MFCC, spektral özellikler)
- Spotify'dan şarkı metadata/popularity bilgisi toplar
- Last.fm'den listeners/playcount alır
- Genius (ve fallback olarak web) ile şarkı sözlerini toplar
- Tüm veriyi `songs.jsonl` ve `songs.csv` içine kaydeder

> Not: Telif, platform kullanım şartları ve veri lisanslarını mutlaka kontrol edin.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

FFmpeg gerekli:

```bash
# macOS
brew install ffmpeg
```

## API Anahtarları

`.env` dosyası oluştur:

```env
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
LASTFM_API_KEY=...
GENIUS_ACCESS_TOKEN=...
```

## Kullanım

```bash
python -m song_miner.main "Blinding Lights" --artist "The Weeknd" --out-dir data
```

Sadece metadata/lyrics çekmek için:

```bash
python -m song_miner.main "Blinding Lights" --artist "The Weeknd" --no-download
```

## Çıktı
- `data/audio/*.wav`
- `data/songs.jsonl`
- `data/songs.csv`

## Model eğitimi için öneri
- `songs.csv` dosyasını okuyup hedef değişken belirleyin (`playcount`, `popularity` veya kendi etiketiniz)
- Özellik vektörü: `mfcc_means + tempo + spectral + lyrics embeddings`
- NLP için sözleri ayrıca tokenize ederek embedding üretin (örn. sentence-transformers)


## Web UI (Modern + Hafif)

Python backend ve hızlı modern arayüz için FastAPI tabanlı web UI eklendi.

Çalıştır:

```bash
uvicorn song_miner.web:app --reload --host 0.0.0.0 --port 8000
```

Aç:
- `http://localhost:8000`

UI üzerinden:
- Şarkı adı / sanatçı gir
- İstersen ses indirmeyi kapat
- Tek tuşla analiz başlat
- JSON sonucu ekranda canlı gör


## Local LLM Agent (Gemma ailesi)

Sisteme gözleyici/yönlendirici/takip edici bir local agent eklendi.

### Ne yapar?
- Pipeline sırasında olayları gözlemler (`agent_observations`)
- Kayıt sonrası veri kalitesi için kısa yönlendirme üretir (`agent_guidance`)
- Gerekirse web'de lyrics araması yapıp LLM ile ayıklamayı dener

### Kurulum (Ollama + Gemma)

```bash
# Ollama kur (macOS)
brew install ollama
ollama serve

# Gemma ailesinden kompakt model çek
ollama pull gemma4:4b
```

Opsiyonel ortam değişkenleri:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
LOCAL_LLM_MODEL=gemma4:4b
```


## Hızlı 3 Aşama (Uygulandı)

1. **Veri kalitesi + dedup**
   - `record_id` üretimi, duplicate kayıt atlama, `quality_confidence` skoru.
2. **Agent takip/gözlem**
   - `agent_events.jsonl` olay kaydı, agent health check endpoint.
3. **Hızlı UI operasyonu**
   - Geçmiş kayıt listesi, kalite skoru/duplicate görünümü, geçmiş yenileme.


## Sonraki Aşamalar (Uygulandı)

- Async job altyapısı eklendi (`/api/analyze_async`, `/api/jobs`, `/api/jobs/{job_id}`)
- UI'ya async analiz butonu ve job polling eklendi
- Agent check + history ile operasyonel takip güçlendirildi


## Yeni 3 Adım (Uygulandı)

1. Job persistence + cancel
   - Job kayıtları `data/jobs.jsonl` içinde tutulur, restart sonrası korunur.
   - `POST /api/jobs/{job_id}/cancel` ile iptal desteği.
2. Spotify eşleşme skoru
   - `spotify_match_score` ve `needs_review` alanları eklendi.
3. UI progress
   - Async job için progress/stage gösterimi ve job iptal butonu eklendi.

## Gemma4
- Varsayılan model `gemma4:4b` olarak güncellendi.


## Agentic Yükseltmeler (Uygulandı)

- Planner: Agent artık JSON plan üretir (`agent_plan`).
- Verifier: Agent kayıtları `accept/retry/review` kararına göre değerlendirir (`agent_verification`).
- Review Queue: `review` kararları `data/review_queue.jsonl` dosyasına atılır ve UI/API'de görüntülenir.
- API: `GET /api/review_queue`


## Agentic Devam (Uygulandı)

- Pipeline için `run_with_hooks(...)` eklendi (stage/progress callback destekli).
- Async job artık gerçek pipeline aşamalarından progress güncelliyor.
- `JobManager.update(...)` ile job stage/progress dışarıdan güncellenebilir hale geldi.
