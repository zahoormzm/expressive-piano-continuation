import base64
import http.client
import json
import threading

import mido
import pytest

from music_project.web.server import (
    BusyError,
    JobManager,
    UserInputError,
    byte_range,
    make_server,
    validate_request,
)


@pytest.fixture
def app_root(tmp_path, midi_builder):
    root = tmp_path / "app"
    (root / "configs").mkdir(parents=True)
    (root / "configs/assets.json").write_text(
        json.dumps(
            {
                "pretrained": {"path": "weights.pt", "source_directory": "aria"},
            }
        )
    )
    (root / "weights.pt").write_bytes(b"fixture, not model weights")
    (root / "aria").mkdir()
    midi = midi_builder(
        [
            (0, mido.Message("note_on", note=60, velocity=80)),
            (500, mido.Message("note_off", note=60)),
        ]
    )
    prompt = root / "outputs/prompts/arpeggio.mid"
    prompt.parent.mkdir(parents=True)
    prompt.write_bytes(midi.read_bytes())
    return root


def test_upload_settings_reject_invalid_seed_numeric_and_payload():
    for body in [
        {"model": "unknown", "use_demo": True},
        {"seed": True, "use_demo": True},
        {"temperature": float("nan"), "use_demo": True},
        {"prompt_seconds": 31, "use_demo": True},
        {"command": "anything", "use_demo": True},
        {"filename": "sample.mid", "midi_base64": "not base64"},
        {"filename": "sample.exe", "midi_base64": "TVRoZA=="},
    ]:
        with pytest.raises(UserInputError):
            validate_request(body)


def test_upload_filename_is_metadata_not_output_path(app_root, monkeypatch):
    manager = JobManager(app_root)
    started = threading.Event()
    monkeypatch.setattr(manager, "_run_job", lambda job_id: started.set())
    content = (app_root / "outputs/prompts/arpeggio.mid").read_bytes()
    job = manager.submit(
        {
            "filename": "../../outside.mid",
            "midi_base64": base64.b64encode(content).decode(),
        }
    )
    assert started.wait(2)
    assert job["filename"] == "outside.mid"
    assert (manager.directory / job["id"] / "input.mid").read_bytes() == content
    assert not (app_root.parent / "outside.mid").exists()
    assert manager.command(job["id"])[0].endswith(("python.exe", "python", "python3"))
    assert "../../outside.mid" not in manager.command(job["id"])


def test_only_one_gpu_job_can_start(app_root, monkeypatch):
    manager = JobManager(app_root)
    started = threading.Event()
    monkeypatch.setattr(manager, "_run_job", lambda job_id: started.set())
    manager.submit({"use_demo": True})
    assert started.wait(2)
    with pytest.raises(BusyError):
        manager.submit({"use_demo": True})


def test_registry_changes_do_not_relabel_existing_samples(app_root, monkeypatch):
    legacy = JobManager(app_root)
    old_id = "b" * 32
    directory = legacy.directory / old_id
    directory.mkdir(parents=True)
    (directory / "job.json").write_text(
        json.dumps({"state": "complete", "settings": {"model": "transformer"}})
    )
    (app_root / "full.pt").write_bytes(b"fixture, not model weights")
    (app_root / "configs/models.json").write_text(
        json.dumps(
            {
                "aria": {"label": "Pretrained Aria", "checkpoint": None},
                "transformer": {"label": "Transformer · 100M tokens", "checkpoint": "full.pt"},
                "lstm": {"label": "LSTM pilot", "checkpoint": "missing.pt"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manager = JobManager(app_root)
    monkeypatch.setattr(manager, "_run_job", lambda job_id: None)
    assert manager.get(old_id)["model_label"] == "Scratch Transformer pilot"
    job = manager.submit({"model": "transformer", "use_demo": True})
    assert job["model_label"] == "Transformer · 100M tokens"
    assert job["checkpoint"] == "full.pt"
    assert manager.command(job["id"])[-1] == "full.pt"
    command = manager.command(job["id"])
    assert command[command.index("--max-new-tokens") + 1] == "auto"


def test_old_fixed_budget_result_reports_actual_continuation_without_rewriting(app_root):
    manager = JobManager(app_root)
    job_id = "c" * 32
    folder = manager.directory / job_id
    (folder / "generation").mkdir(parents=True)
    job = {"state": "complete", "settings": {"model": "transformer"}, "result": {}}
    stored = json.dumps(job)
    (folder / "job.json").write_text(stored)
    report = {"audio": {"continuation": {"seconds": 11.6}}}
    (folder / "generation/report.json").write_text(json.dumps(report))
    result = manager.get(job_id)
    assert result["model_label"] == "Scratch Transformer pilot"
    assert result["result"]["continuation_audio_seconds"] == 11.6
    assert result["result"]["generation_token_budget"] == 384
    assert result["result"]["generation_budget_mode"] == "fixed"
    assert (folder / "job.json").read_text() == stored
    report["generation_limits"] = {"max_new_tokens": 1354, "budget_mode": "remaining_context"}
    (folder / "generation/report.json").write_text(json.dumps(report))
    assert manager.get(job_id)["result"]["generation_token_budget"] == 1354


def test_nonpiano_and_malformed_uploads_never_start_worker(app_root, midi_builder, monkeypatch):
    manager = JobManager(app_root)
    started = []
    monkeypatch.setattr(manager, "_run_job", lambda job_id: started.append(job_id))
    nonpiano = midi_builder(
        [
            (0, mido.Message("program_change", program=40)),
            (0, mido.Message("note_on", note=60, velocity=80)),
            (500, mido.Message("note_off", note=60)),
        ],
        "violin.mid",
    )
    for content in [nonpiano.read_bytes(), b"MThd" + b"\x00" * 30]:
        with pytest.raises(UserInputError):
            manager.submit(
                {"filename": "example.mid", "midi_base64": base64.b64encode(content).decode()}
            )
    assert started == [] and manager.active is None


def test_restarted_incomplete_job_is_reported_as_failed(app_root, monkeypatch):
    manager = JobManager(app_root)
    monkeypatch.setattr(manager, "_run_job", lambda job_id: None)
    job = manager.submit({"use_demo": True})
    restarted = JobManager(app_root)
    assert restarted.get(job["id"])["state"] == "failed"
    assert "server stopped" in restarted.get(job["id"])["error"]


def test_audio_ranges_support_seeking_and_reject_invalid_offsets():
    assert byte_range(None, 100) == (0, 99, False)
    assert byte_range("bytes=10-19", 100) == (10, 19, True)
    assert byte_range("bytes=80-", 100) == (80, 99, True)
    assert byte_range("bytes=-12", 100) == (88, 99, True)
    assert byte_range("bytes=0-1000", 100) == (0, 99, True)
    for header in ["bytes=100-", "bytes=20-10", "bytes=-0", "bytes=", "bytes=1-2,4-5"]:
        with pytest.raises(ValueError):
            byte_range(header, 100)


def test_http_host_origin_artifact_boundary_and_audio_range(app_root):
    server = make_server(app_root, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    manager = server.manager
    job_id = "a" * 32
    directory = manager.directory / job_id
    (directory / "generation").mkdir(parents=True)
    (directory / "job.json").write_text(json.dumps({"state": "complete"}))
    audio = bytes(range(100))
    (directory / "generation/combined.wav").write_bytes(audio)

    def request(path, headers=None, method="GET", body=None):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    try:
        assert request("/")[0] == 200
        assert request("/api/health", {"Host": "external.example"})[0] == 403
        assert (
            request(
                "/api/jobs",
                {"Origin": "https://external.example", "Content-Type": "application/json"},
                "POST",
                '{"use_demo":true}',
            )[0]
            == 403
        )
        status, headers, data = request(
            f"/artifacts/{job_id}/combined.wav", {"Range": "bytes=10-19"}
        )
        assert status == 206 and data == audio[10:20]
        assert headers["Content-Range"] == "bytes 10-19/100"
        assert request(f"/artifacts/{job_id}/combined.wav", {"Range": "bytes=100-"})[0] == 416
        assert request(f"/artifacts/{job_id}/worker.log")[0] == 404
        assert request("/artifacts/../weights.pt")[0] == 404
        assert request("/api/jobs/" + "b" * 32)[0] == 404
        assert request("/api/jobs", {"Content-Type": "text/plain"}, "POST", "{}")[0] == 415
        assert (
            request(
                "/api/jobs",
                {"Content-Type": "application/json"},
                "POST",
                '{"seed":-1,"use_demo":true}',
            )[0]
            == 400
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
