from __future__ import annotations

import csv
import json
import hashlib
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Optional

import librosa
import numpy as np
import requests
import spotipy
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from lyricsgenius import Genius
from spotipy.oauth2 import SpotifyClientCredentials
from yt_dlp import YoutubeDL

from song_miner.agent import LocalLLMAgent


@dataclass
class SongRecord:
    query: str
    artist: Optional[str]
    resolved_title: Optional[str]
    resolved_artist: Optional[str]
    spotify_id: Optional[str]
    spotify_popularity: Optional[int]
    spotify_duration_ms: Optional[int]
    spotify_rank_in_search: Optional[int]
    lastfm_listeners: Optional[int]
    lastfm_playcount: Optional[int]
    lyrics: Optional[str]
    audio_path: Optional[str]
    features: dict[str, Any]
    collected_at_utc: str


class SongPipeline:
    def __init__(self, out_dir: str = "data") -> None:
        load_dotenv()
        self.out_dir = Path(out_dir)
        self.agent = LocalLLMAgent(out_dir=out_dir)
        self.seen_ids_path = self.out_dir / "seen_ids.txt"
        self.audio_dir = self.out_dir / "audio"
        self.review_queue_path = self.out_dir / "review_queue.jsonl"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)


    def run(
        self,
        query: str,
        artist: Optional[str] = None,
        max_seconds: int = 120,
        skip_download: bool = False,
    ) -> dict[str, Any]:
        return self.run_with_hooks(
            query=query,
            artist=artist,
            max_seconds=max_seconds,
            skip_download=skip_download,
            progress_cb=None,
        )

    def run_with_hooks(
        self,
        query: str,
        artist: Optional[str] = None,
        max_seconds: int = 120,
        skip_download: bool = False,
        progress_cb=None,
    ) -> dict[str, Any]:
        def _p(stage: str, progress: int) -> None:
            if callable(progress_cb):
                try:
                    progress_cb(stage, progress)
                except Exception:
                    pass

        _p("starting", 5)
        self.agent.observe("run_start", f"query={query} artist={artist}")
        spotify_meta = self._spotify_lookup(query, artist)
        title = spotify_meta.get("title")
        resolved_artist = spotify_meta.get("artist") or artist
        _p("metadata", 20)

        lastfm_meta = self._lastfm_stats(title or query, resolved_artist)
        lyrics = self._fetch_lyrics(title or query, resolved_artist)
        if not lyrics:
            lyrics = self.agent.fetch_lyrics_via_web(title or query, resolved_artist)
        _p("lyrics", 45)

        audio_path = None
        features: dict[str, Any] = {}
        if not skip_download:
            audio_path = self._download_audio(title or query, resolved_artist)
            if audio_path:
                features = self._extract_audio_features(audio_path, max_seconds=max_seconds)
        _p("audio_features", 75)

        confidence = self._confidence_score(spotify_meta, lastfm_meta, lyrics, audio_path, features)

        record = SongRecord(
            query=query,
            artist=artist,
            resolved_title=title,
            resolved_artist=resolved_artist,
            spotify_id=spotify_meta.get("id"),
            spotify_popularity=spotify_meta.get("popularity"),
            spotify_duration_ms=spotify_meta.get("duration_ms"),
            spotify_rank_in_search=spotify_meta.get("rank"),
            lastfm_listeners=lastfm_meta.get("listeners"),
            lastfm_playcount=lastfm_meta.get("playcount"),
            lyrics=lyrics,
            audio_path=audio_path,
            features=features,
            collected_at_utc=datetime.now(timezone.utc).isoformat(),
        )

        rec = asdict(record)
        rec["quality_confidence"] = confidence
        rec["spotify_match_score"] = spotify_meta.get("match_score")
        rec["needs_review"] = bool((spotify_meta.get("match_score") or 0) < 0.65)
        rec["record_id"] = self._record_id(query, artist, title, resolved_artist)
        rec["agent_observations"] = self.agent.export_observations()
        rec["agent_guidance"] = self.agent.summarize_guidance(rec)
        rec["agent_plan"] = self.agent.plan_json("collect high-quality song training record", {"query": query, "artist": artist})

        verification = self.agent.verify_record(rec)
        rec["agent_verification"] = verification
        rec["retried_once"] = verification.get("decision") == "retry"
        if rec["retried_once"]:
            self.agent.observe("retry_once", rec["record_id"])

        if verification.get("decision") == "review":
            self._append_jsonl(self.review_queue_path, rec)

        if self._is_duplicate(rec["record_id"]):
            rec["duplicate"] = True
            self.agent.observe("duplicate_skip", rec["record_id"])
            _p("duplicate", 100)
            return rec

        rec["duplicate"] = False
        self._remember_id(rec["record_id"])
        self._append_jsonl(self.out_dir / "songs.jsonl", rec)
        self._append_csv(self.out_dir / "songs.csv", rec)
        _p("stored", 100)
        return rec

    def _spotify_lookup(self, query: str, artist: Optional[str]) -> dict[str, Any]:
        cid = os.getenv("SPOTIFY_CLIENT_ID")
        sec = os.getenv("SPOTIFY_CLIENT_SECRET")
        if not cid or not sec:
            return {}

        auth = SpotifyClientCredentials(client_id=cid, client_secret=sec)
        sp = spotipy.Spotify(auth_manager=auth)
        q = f"track:{query}" + (f" artist:{artist}" if artist else "")
        res = sp.search(q=q, type="track", limit=10)
        items = res.get("tracks", {}).get("items", [])
        if not items:
            return {}
        first = items[0]
        track_name = first.get("name") or ""
        artist_name = first.get("artists", [{}])[0].get("name") or ""
        q_score = SequenceMatcher(None, query.lower(), track_name.lower()).ratio()
        a_score = SequenceMatcher(None, (artist or "").lower(), artist_name.lower()).ratio() if artist else 1.0
        match_score = round((q_score * 0.7 + a_score * 0.3), 3)
        return {
            "id": first.get("id"),
            "title": first.get("name"),
            "artist": first.get("artists", [{}])[0].get("name"),
            "popularity": first.get("popularity"),
            "duration_ms": first.get("duration_ms"),
            "rank": 1,
            "match_score": match_score,
        }

    def _lastfm_stats(self, title: str, artist: Optional[str]) -> dict[str, Optional[int]]:
        key = os.getenv("LASTFM_API_KEY")
        if not key:
            return {"listeners": None, "playcount": None}

        params = {
            "method": "track.getInfo",
            "api_key": key,
            "format": "json",
            "track": title,
        }
        if artist:
            params["artist"] = artist

        r = requests.get("https://ws.audioscrobbler.com/2.0/", params=params, timeout=20)
        r.raise_for_status()
        data = r.json().get("track", {})
        return {
            "listeners": int(data["listeners"]) if data.get("listeners") else None,
            "playcount": int(data["playcount"]) if data.get("playcount") else None,
        }

    def _fetch_lyrics(self, title: str, artist: Optional[str]) -> Optional[str]:
        genius_token = os.getenv("GENIUS_ACCESS_TOKEN")
        if genius_token:
            genius = Genius(genius_token, verbose=False, remove_section_headers=True)
            song = genius.search_song(title=title, artist=artist)
            if song and song.lyrics:
                return song.lyrics

        # Fallback: azlyrics scrape (best-effort, kırılgan olabilir)
        try:
            q = f"{title} {artist or ''} lyrics"
            url = f"https://duckduckgo.com/html/?q={requests.utils.quote(q)}"
            html = requests.get(url, timeout=20).text
            soup = BeautifulSoup(html, "html.parser")
            links = [a.get("href") for a in soup.select("a.result__a") if a.get("href")]
            for link in links:
                if "azlyrics.com/lyrics/" in link:
                    page = requests.get(link, timeout=20).text
                    s2 = BeautifulSoup(page, "html.parser")
                    divs = s2.find_all("div")
                    for div in divs:
                        txt = div.get_text("\n", strip=True)
                        if txt and len(txt.split()) > 40:
                            return txt
        except Exception:
            return None
        return None

    def _download_audio(self, title: str, artist: Optional[str]) -> Optional[str]:
        search_q = f"ytsearch1:{title} {artist or ''} audio"
        outtmpl = str(self.audio_dir / "%(title)s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "noplaylist": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_q, download=True)
            entries = info.get("entries", []) if info else []
            if not entries:
                return None
            ent = entries[0]
            filepath = self.audio_dir / f"{ent.get('title')}.wav"
            return str(filepath) if filepath.exists() else None

    def _extract_audio_features(self, path: str, max_seconds: int = 120) -> dict[str, Any]:
        y, sr = librosa.load(path, sr=None, mono=True, duration=max_seconds)
        if y.size == 0:
            return {}

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        rms = float(np.mean(librosa.feature.rms(y=y)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = [float(np.mean(m)) for m in mfcc]

        return {
            "sample_rate": sr,
            "duration_sec": float(len(y) / sr),
            "tempo_bpm": float(tempo),
            "zcr_mean": zcr,
            "rms_mean": rms,
            "spectral_centroid_mean": centroid,
            "spectral_bandwidth_mean": bandwidth,
            "spectral_rolloff_mean": rolloff,
            "mfcc_means": mfcc_means,
        }

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _append_csv(path: Path, record: dict[str, Any]) -> None:
        flat = record.copy()
        flat["features"] = json.dumps(flat.get("features", {}), ensure_ascii=False)

        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(flat)

    def _record_id(self, query: str, artist: Optional[str], title: Optional[str], resolved_artist: Optional[str]) -> str:
        raw = "|".join([query or "", artist or "", title or "", resolved_artist or ""]).lower().strip()
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _is_duplicate(self, record_id: str) -> bool:
        if not self.seen_ids_path.exists():
            return False
        with self.seen_ids_path.open("r", encoding="utf-8") as f:
            return record_id in {line.strip() for line in f if line.strip()}

    def _remember_id(self, record_id: str) -> None:
        with self.seen_ids_path.open("a", encoding="utf-8") as f:
            f.write(record_id + "\n")

    def _confidence_score(self, spotify_meta: dict[str, Any], lastfm_meta: dict[str, Optional[int]], lyrics: Optional[str], audio_path: Optional[str], features: dict[str, Any]) -> float:
        score = 0.0
        if spotify_meta.get("id"):
            score += 0.3
        if lastfm_meta.get("listeners") is not None:
            score += 0.2
        if lyrics and len(lyrics.split()) > 40:
            score += 0.2
        if audio_path:
            score += 0.15
        if features:
            score += 0.15
        return round(min(score, 1.0), 2)

