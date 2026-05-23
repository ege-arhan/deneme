from song_miner.jobs import JobManager

def test_job_lifecycle(tmp_path):
    jm = JobManager(data_dir=str(tmp_path))
    job = jm.create()
    assert job.status == "queued"
    jm.update(job.id, stage="processing", progress=30)
    loaded = jm.get(job.id)
    assert loaded.stage == "processing"
    assert loaded.progress == 30
    assert jm.cancel(job.id) is True
