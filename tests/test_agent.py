from song_miner.agent import LocalLLMAgent

def test_verify_and_action(tmp_path):
    agent = LocalLLMAgent(out_dir=str(tmp_path))
    rec = {"needs_review": True, "lyrics": None, "features": {}}
    ver = agent.verify_record(rec)
    assert ver["decision"] in {"retry", "review", "accept"}
    act = agent.decide_next_action({"agent_verification": ver})
    assert "action" in act
