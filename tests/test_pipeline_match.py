from song_miner.pipeline import SongPipeline

def test_confidence_score_uses_match():
    p = SongPipeline(out_dir='data_test_tmp')
    sc = p._confidence_score({"id":"x","match_score":0.5}, {"listeners":1}, "hello world "*50, "a.wav", {"x":1})
    assert 0 <= sc <= 1
