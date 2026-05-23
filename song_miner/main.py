import argparse
from song_miner.pipeline import SongPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Şarkı indir, analiz et, istatistik + söz topla ve diske kaydet"
    )
    parser.add_argument("query", help="Şarkı adı veya 'sanatçı - şarkı' formatı")
    parser.add_argument("--artist", default=None, help="Sanatçı adı (opsiyonel)")
    parser.add_argument("--out-dir", default="data", help="Çıktı klasörü")
    parser.add_argument(
        "--audio-seconds",
        type=int,
        default=120,
        help="Analiz için indirilecek maksimum saniye",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Sadece metadata/statistics/söz çek; ses indirme",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = SongPipeline(out_dir=args.out_dir)
    result = pipeline.run(
        query=args.query,
        artist=args.artist,
        max_seconds=args.audio_seconds,
        skip_download=args.no_download,
    )
    print("\nTamamlandı. Kayıt:")
    print(result)


if __name__ == "__main__":
    main()
