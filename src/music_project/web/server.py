"""Loopback-only MIDI jobs, isolated GPU processes, and bounded artifact serving."""

import base64
import copy
import json
import math
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from music_project.data.midi import read_performance
from music_project.io import write_bytes, write_json
from music_project.resources import gpu_status

MAX_MIDI_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BYTES = 3 * 1024 * 1024
MODELS = {
    "aria": ("Pretrained Aria", None),
    "transformer": (
        "Scratch Transformer pilot",
        "runs/transformer-small-pilot-20261006-001/best.pt",
    ),
    "lstm": ("LSTM pilot", "runs/lstm-pilot-20261006-001/best.pt"),
}
FILES = {
    "prompt.mid": "audio/midi",
    "combined.mid": "audio/midi",
    "continuation.mid": "audio/midi",
    "prompt.wav": "audio/wav",
    "combined.wav": "audio/wav",
    "continuation.wav": "audio/wav",
    "report.json": "application/json",
}
JOB_ID = re.compile(r"[a-f0-9]{32}\Z")


class UserInputError(ValueError):
    pass


class BusyError(RuntimeError):
    pass


def validate_request(body):
    fields = {
        "model",
        "seed",
        "prompt_seconds",
        "new_seconds",
        "temperature",
        "min_p",
        "use_demo",
        "filename",
        "midi_base64",
    }
    if not isinstance(body, dict) or set(body) - fields:
        raise UserInputError("Unknown generation settings.")
    model = body.get("model", "aria")
    if not isinstance(model, str) or model not in MODELS:
        raise UserInputError("Select an available piano model.")
    settings = {"model": model}
    for field, default, low, high in [
        ("prompt_seconds", 8, 1, 30),
        ("new_seconds", 12, 1, 30),
        ("temperature", 0.98, 0, 1.5),
        ("min_p", 0.035, 0, 1),
    ]:
        value = body.get(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UserInputError(f"{field.replace('_', ' ')} must be a number.")
        if not math.isfinite(value) or not low <= value <= high:
            raise UserInputError(f"{field.replace('_', ' ')} must be between {low} and {high}.")
        settings[field] = value
    seed = body.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise UserInputError("Seed must be a whole number between 0 and 4294967295.")
    settings["seed"] = seed
    demo = body.get("use_demo", False)
    if not isinstance(demo, bool):
        raise UserInputError("Choose the demo or a MIDI file.")
    if demo:
        return settings, None, "Demo arpeggio.mid"
    name, encoded = body.get("filename"), body.get("midi_base64")
    if (
        not isinstance(name, str)
        or len(name) > 160
        or not name.lower().endswith((".mid", ".midi"))
        or any(ord(character) < 32 for character in name)
    ):
        raise UserInputError("Choose a .mid or .midi file.")
    if not isinstance(encoded, str) or len(encoded) > (MAX_MIDI_BYTES + 2) // 3 * 4:
        raise UserInputError("MIDI files must be at most 2 MB.")
    try:
        midi = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise UserInputError("The MIDI upload could not be decoded.") from exc
    if not 14 <= len(midi) <= MAX_MIDI_BYTES or midi[:4] != b"MThd":
        raise UserInputError("This file is not a supported MIDI file.")
    # The display name is metadata only. Uploaded bytes always use input.mid.
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    return settings, midi, name


def validate_midi(path, prompt_seconds):
    try:
        performance = read_performance(path)
    except Exception as exc:
        raise UserInputError(
            "The MIDI could not be read; use a synchronous piano MIDI file."
        ) from exc
    if len(performance.notes) > 10_000 or performance.duration_seconds > 600:
        raise UserInputError("Use a piano excerpt of at most ten minutes and 10,000 notes.")
    if any(note.channel == 9 or not 0 <= note.program <= 7 for note in performance.notes):
        raise UserInputError("Use solo-piano MIDI without drums or other instruments.")
    if any(not 21 <= note.pitch <= 108 for note in performance.notes):
        raise UserInputError("Use notes within the piano range, A0 to C8.")
    if any(
        performance.issues.get(key)
        for key in (
            "unmatched_note_on",
            "unmatched_note_off",
            "nonpositive_note_duration",
        )
    ):
        raise UserInputError(
            "The MIDI has unfinished or invalid notes; repair it before uploading."
        )
    if not any(note.start_seconds < prompt_seconds for note in performance.notes):
        raise UserInputError("There are no notes in the selected prompt duration.")


class JobManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        registry = self.root / "configs/models.json"
        self.models = (
            {
                key: (value["label"], value["checkpoint"])
                for key, value in json.loads(registry.read_text(encoding="utf-8")).items()
            }
            if registry.exists()
            else MODELS
        )
        if set(self.models) != set(MODELS):
            raise ValueError("Model registry must contain aria, transformer and lstm")
        self.directory = self.root / "outputs" / "app-jobs"
        self.lock = threading.RLock()
        self.active = None
        self.process = None

    def available_models(self):
        assets = json.loads((self.root / "configs/assets.json").read_text())
        result = []
        for model, (label, checkpoint) in self.models.items():
            paths = (
                [self.root / checkpoint]
                if checkpoint
                else [
                    self.root / assets["pretrained"]["path"],
                    self.root / assets["pretrained"]["source_directory"],
                ]
            )
            result.append(
                {"id": model, "label": label, "available": all(p.exists() for p in paths)}
            )
        return result

    def _status_path(self, job_id):
        if not JOB_ID.fullmatch(job_id):
            raise KeyError("Unknown generation.")
        return self.directory / job_id / "job.json"

    def get(self, job_id):
        with self.lock:
            path = self._status_path(job_id)
            if not path.exists():
                raise KeyError("Unknown generation.")
            job = json.loads(path.read_text(encoding="utf-8"))
            # Results made before the registry existed retain their original pilot label.
            if "settings" in job and "model_label" not in job:
                job["model_label"] = MODELS[job["settings"]["model"]][0]
            report_path = path.parent / "generation/report.json"
            if job["state"] == "complete" and "result" in job and report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                limits = report.get("generation_limits", {})
                job["result"].update(
                    continuation_audio_seconds=report["audio"]["continuation"]["seconds"],
                    generation_token_budget=limits.get("max_new_tokens", 384),
                    generation_budget_mode=limits.get("budget_mode", "fixed"),
                )
            if (
                job["state"] in ("validating", "loading", "generating", "rendering")
                and self.active != job_id
            ):
                job.update(
                    state="failed",
                    error="The server stopped before this generation completed. Start a new generation.",
                )
            return job

    def _update(self, job_id, **changes):
        with self.lock:
            path = self._status_path(job_id)
            job = json.loads(path.read_text(encoding="utf-8"))
            job.update(changes)
            write_json(path, job)

    def submit(self, body):
        settings, content, filename = validate_request(body)
        with self.lock:
            if self.active is not None:
                raise BusyError("A generation is already running. Please wait for it to finish.")
            resource = gpu_status(self.root)
            if resource["busy"]:
                raise BusyError(
                    f"GPU is busy with {resource['activity']}. Please wait for it to finish."
                )
            if not any(
                m["id"] == settings["model"] and m["available"] for m in self.available_models()
            ):
                raise UserInputError(
                    "The selected model is missing. See the project setup instructions."
                )
            job_id = uuid.uuid4().hex
            directory = self.directory / job_id
            directory.mkdir(parents=True, exist_ok=False)
            path = directory / "input.mid"
            if content is None:
                demo = self.root / "outputs/prompts/arpeggio.mid"
                if not demo.exists():
                    raise UserInputError(
                        "Demo MIDI is missing. Run scripts/make_demo_prompt.py first."
                    )
                content = demo.read_bytes()
            write_bytes(path, content, immutable=True)
            job = {
                "id": job_id,
                "state": "validating",
                "filename": filename,
                "settings": settings,
                "model_label": self.models[settings["model"]][0],
                "checkpoint": self.models[settings["model"]][1],
                "created_unix": time.time(),
                "error": None,
            }
            write_json(directory / "job.json", job)
            try:
                validate_midi(path, settings["prompt_seconds"])
            except UserInputError as exc:
                self._update(job_id, state="failed", error=str(exc))
                raise
            self.active = job_id
            threading.Thread(target=self._run_job, args=(job_id,), daemon=True).start()
            return copy.deepcopy(job)

    def command(self, job_id):
        job = self.get(job_id)
        settings = job["settings"]
        directory = self.directory / job_id
        command = [
            sys.executable,
            "-u",
            str(self.root / "scripts/generate.py"),
            "--prompt",
            str(directory / "input.mid"),
            "--output",
            str(directory / "generation"),
            "--render",
            "--prompt-seconds",
            str(settings["prompt_seconds"]),
            "--new-seconds",
            str(settings["new_seconds"]),
            "--temperature",
            str(settings["temperature"]),
            "--min-p",
            str(settings["min_p"]),
            "--seed",
            str(settings["seed"]),
            "--max-new-tokens",
            "auto",
        ]
        if settings["model"] == "aria":
            command.extend(["--backend", "aria"])
        else:
            command.extend(
                ["--checkpoint", job.get("checkpoint", self.models[settings["model"]][1])]
            )
        return command

    def _run_job(self, job_id):
        directory = self.directory / job_id
        try:
            self._update(job_id, state="loading")
            with (directory / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    self.command(job_id),
                    cwd=self.root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                with self.lock:
                    self.process = process
                timer = threading.Timer(600, process.kill)
                timer.start()
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith("PROGRESS\t"):
                            progress = json.loads(line.partition("\t")[2])
                            if progress["stage"] in ("loading", "generating", "rendering"):
                                self._update(job_id, state=progress["stage"])
                    code = process.wait()
                finally:
                    timer.cancel()
            if code != 0:
                detail = (directory / "worker.log").read_text(encoding="utf-8").splitlines()
                last = next(
                    (line for line in reversed(detail) if line.strip()), "Generation failed."
                )
                if "context" in last.lower():
                    last = "This prompt exceeds the model context. Use a shorter prompt or less dense MIDI."
                elif "No saved continuation notes" in last:
                    last = "No continuation notes were produced. Try another seed or model."
                elif "clips at the fixed gain" in last:
                    last = (
                        "The piano audio clipped. This result was not accepted; try another seed."
                    )
                raise RuntimeError(last[:500])
            report = json.loads((directory / "generation/report.json").read_text(encoding="utf-8"))
            if (
                not report["export"]["prompt_preserved"]
                or not report["export"]["note_count_preserved"]
            ):
                raise RuntimeError("The MIDI integrity checks did not pass.")
            artifacts = {
                name: f"/artifacts/{job_id}/{name}"
                for name in FILES
                if (directory / "generation" / name).is_file()
            }
            self._update(
                job_id,
                state="complete",
                completed_unix=time.time(),
                artifacts=artifacts,
                result={
                    "generated_notes": report["export"]["continuation"]["notes"],
                    "sampling_seconds": report["sampling_seconds"],
                    "prompt_notes": report["prompt"]["prompt_notes"],
                    "stop_reason": report["sampling"]["stop_reason"],
                    "combined_audio_seconds": report["audio"]["combined"]["seconds"],
                    "continuation_audio_seconds": report["audio"]["continuation"]["seconds"],
                    "generation_token_budget": report["generation_limits"]["max_new_tokens"],
                    "generation_budget_mode": report["generation_limits"]["budget_mode"],
                    "note_end_seconds": report["export"]["continuation_note_end_seconds"],
                    "prompt_preserved": True,
                    "note_count_preserved": True,
                },
            )
        except Exception as exc:  # noqa: BLE001 -- publish worker failures to the local job
            self._update(job_id, state="failed", error=str(exc)[:500])
        finally:
            with self.lock:
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                self.process = None
                self.active = None

    def artifact(self, job_id, filename):
        job = self.get(job_id)
        if job["state"] != "complete" or filename not in FILES:
            raise KeyError("Unknown artifact.")
        path = self.directory / job_id / "generation" / filename
        if not path.is_file() or not path.resolve().is_relative_to(self.directory.resolve()):
            raise KeyError("Unknown artifact.")
        return path, FILES[filename]

    def close(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()


def byte_range(header, length):
    if header is None:
        return 0, length - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()):
        raise ValueError("Invalid byte range")
    first, last = match.groups()
    if first:
        start, end = int(first), min(int(last), length - 1) if last else length - 1
    else:
        count = int(last)
        if count <= 0:
            raise ValueError("Invalid suffix range")
        start, end = max(0, length - count), length - 1
    if not 0 <= start <= end < length:
        raise ValueError("Unsatisfiable byte range")
    return start, end, True


def make_server(root, port=8767):
    manager = JobManager(root)
    page = Path(__file__).with_name("index.html")

    class Handler(BaseHTTPRequestHandler):
        def _local_request(self):
            port = self.server.server_port
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in hosts:
                self._json(403, {"error": "Open this application from its local address."})
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin not in {f"http://{host}" for host in hosts}:
                self._json(
                    403, {"error": "Generation requests must come from this local application."}
                )
                return False
            return True

        def _headers(self, code, content_type, length, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()

        def _json(self, code, body):
            content = json.dumps(body, allow_nan=False).encode("utf-8")
            self._headers(code, "application/json; charset=utf-8", len(content))
            self.wfile.write(content)

        def do_GET(self):
            if not self._local_request():
                return
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    content = page.read_bytes()
                    self._headers(200, "text/html; charset=utf-8", len(content))
                    self.wfile.write(content)
                elif path == "/api/health":
                    resource = gpu_status(manager.root)
                    self._json(
                        200,
                        {
                            "models": manager.available_models(),
                            "busy": manager.active is not None or resource["busy"],
                            "activity": resource["activity"],
                        },
                    )
                elif path.startswith("/api/jobs/") and path.count("/") == 3:
                    self._json(200, manager.get(path.rsplit("/", 1)[1]))
                elif path.startswith("/artifacts/") and path.count("/") == 3:
                    _, _, job_id, filename = path.split("/")
                    artifact, mime = manager.artifact(job_id, filename)
                    length = artifact.stat().st_size
                    try:
                        start, end, ranged = byte_range(self.headers.get("Range"), length)
                    except ValueError:
                        self._headers(416, mime, 0, {"Content-Range": f"bytes */{length}"})
                        return
                    headers = {"Accept-Ranges": "bytes"}
                    if ranged:
                        headers["Content-Range"] = f"bytes {start}-{end}/{length}"
                    if artifact.suffix in (".mid", ".json"):
                        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
                    self._headers(206 if ranged else 200, mime, end - start + 1, headers)
                    with artifact.open("rb") as source:
                        source.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = source.read(min(remaining, 64 * 1024))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                else:
                    self._json(404, {"error": "Page not found."})
            except KeyError:
                self._json(404, {"error": "Generation or artifact not found."})
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if not self._local_request():
                return
            if urlsplit(self.path).path != "/api/jobs":
                self._json(404, {"error": "Page not found."})
                return
            if self.headers.get("Content-Type", "").partition(";")[0] != "application/json":
                self._json(415, {"error": "Use the MIDI upload form."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    self._json(
                        413, {"error": "The upload is too large; use a MIDI file under 2 MB."}
                    )
                    return
                self.connection.settimeout(15)
                body = json.loads(self.rfile.read(length))
                self._json(202, manager.submit(body))
            except BusyError as exc:
                self._json(409, {"error": str(exc)})
            except (ValueError, UnicodeError) as exc:
                self._json(
                    400,
                    {
                        "error": str(exc)
                        if isinstance(exc, UserInputError)
                        else "The upload request could not be read."
                    },
                )

        def log_message(self, format, *args):
            print(f"{self.address_string()} {format % args}", flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.manager = manager
    return server
