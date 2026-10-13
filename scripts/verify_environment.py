"""Record the actual Windows/CUDA environment and execute GPU forward/backward."""

import argparse
import importlib.metadata
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from music_project.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/environment.json"))
    args = parser.parse_args()
    report = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            package.metadata["Name"]: package.version
            for package in importlib.metadata.distributions()
        },
    }
    try:
        report["nvidia_smi"] = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        ).stdout
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this environment")
        torch.manual_seed(42)
        x = torch.randn(512, 512, device="cuda", requires_grad=True)
        loss = (x @ x.T).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss).item() or not torch.isfinite(x.grad).all().item():
            raise RuntimeError("GPU forward/backward produced nonfinite values")
        report.update(
            {
                "torch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "forward_backward_passed": True,
                "loss": loss.item(),
            }
        )
        if report["bf16_supported"]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mixed_loss = (x @ x.T).square().mean()
            mixed_loss.backward()
            torch.cuda.synchronize()
            if not torch.isfinite(mixed_loss).item() or not torch.isfinite(x.grad).all().item():
                raise RuntimeError("BF16 GPU forward/backward produced nonfinite values")
            report["bf16_forward_backward_passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(args.output, report)
        raise
    write_json(args.output, report)
    print(
        f"{report['gpu']} | torch {report['torch']} | CUDA {report['cuda_build']} | "
        f"BF16 {report['bf16_supported']} | GPU forward/backward passed"
    )


if __name__ == "__main__":
    main()
