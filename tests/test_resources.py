import subprocess
import sys

import pytest

from music_project.resources import ResourceBusyError, gpu_slot, gpu_status
from music_project.web.server import BusyError, JobManager


def test_gpu_slot_excludes_other_processes_and_releases_after_failure(tmp_path):
    code = (
        "import sys; from music_project.resources import gpu_slot; "
        "slot=gpu_slot('generation',root=sys.argv[1]); slot.__enter__()"
    )
    with pytest.raises(ValueError, match="deliberate failure"), gpu_slot("training", root=tmp_path):
        assert gpu_status(tmp_path) == {"busy": True, "activity": "training"}
        other = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert other.returncode != 0 and "GPU is busy with training" in other.stderr
        with pytest.raises(ResourceBusyError), gpu_slot("generation", root=tmp_path):
            pass
        raise ValueError("deliberate failure")
    assert gpu_status(tmp_path) == {"busy": False, "activity": None}


def test_application_rejects_generation_while_training_owns_gpu(tmp_path):
    manager = JobManager(tmp_path)
    with gpu_slot("training", root=tmp_path), pytest.raises(BusyError, match="training"):
        manager.submit({"use_demo": True})
    assert manager.active is None
