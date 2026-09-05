"""Environment sanity report for RetinaPrep.

Run as `python -m scripts.doctor` from the repo root, or directly:
`.\\.venv\\Scripts\\python.exe scripts\\doctor.py`

Exits 0 if everything checked out (torch simply being absent is not a
problem -- only `train` and dedupe's embedding path need it). Exits 1 if a
real problem was found: a GPU that torch can see but cannot actually
compute on. Two ways that happens, both checked below:
  - the installed wheel targets a newer CUDA than the driver supports
    (compares torch.version.cuda against the ceiling nvidia-smi reports)
  - the wheel dropped this machine's compute capability (Pascal / sm_61)
    while still reporting cuda.is_available() == True
Either is a silent failure -- detection succeeding proves nothing about
kernels actually running -- which is exactly what the version check, the
get_arch_list check, and the matmul below exist to catch.

Whether Python resolved outside .venv, and whether the raw dataset has
landed yet, are reported but do not affect the exit code: both are normal
states at various points in this project (checking from a different
interpreter on purpose; not having downloaded ODIR-5K yet).
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ARCH = "sm_61"  # Pascal, e.g. this project's GTX 1050 dev box.

problems: list[str] = []


def section(title: str) -> None:
    print(f"\n== {title} ==")


def nvidia_smi_gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip().splitlines()[0]


def nvidia_smi_max_cuda_version() -> str | None:
    """The driver's CUDA capability ceiling, e.g. "12.9".

    This is not exposed as a --query-gpu field (it's a value nvidia-smi
    derives from the driver, not a GPU property), so it has to be scraped
    from the plain-text header it prints, e.g.
    "...Driver Version: 576.88         CUDA Version: 12.9...".
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
    return match.group(1) if match else None


def check_python() -> None:
    section("Python")
    print(f"  version: {sys.version.split()[0]}")
    print(f"  executable: {sys.executable}")

    venv_dir = (REPO_ROOT / ".venv").resolve()
    in_project_venv = Path(sys.prefix).resolve() == venv_dir
    if in_project_venv:
        print(f"  inside project .venv: yes ({venv_dir})")
    else:
        print(f"  inside project .venv: NO (expected {venv_dir})")
        print("  WARNING: not running from the project .venv -- results below describe")
        print("           whatever interpreter this is, not necessarily the project env.")


def check_torch() -> None:
    section("torch")
    try:
        import torch
    except ImportError:
        print("  not installed (fine unless you're about to run train / dedupe embeddings)")
        return

    version = torch.__version__
    if "+cpu" in version:
        build_kind = "cpu"
    elif "+cu" in version:
        build_kind = version.split("+", 1)[1]
    else:
        build_kind = "unknown (no +cpu/+cuXXX suffix)"
    print(f"  version: {version}  (build: {build_kind})")

    driver_max_cuda = nvidia_smi_max_cuda_version()
    wheel_cuda = torch.version.cuda  # e.g. "12.6"; None for a +cpu build
    if driver_max_cuda:
        print(f"  driver's max supported CUDA (nvidia-smi): {driver_max_cuda}")
    if wheel_cuda and driver_max_cuda:
        try:
            wheel_v = tuple(int(p) for p in wheel_cuda.split(".")[:2])
            driver_v = tuple(int(p) for p in driver_max_cuda.split(".")[:2])
        except ValueError:
            wheel_v = driver_v = None  # unparsable version string; don't guess
        if wheel_v is not None and wheel_v > driver_v:
            problems.append(
                f"torch was built for CUDA {wheel_cuda} but the installed driver only "
                f"supports up to CUDA {driver_max_cuda} (per nvidia-smi). "
                "cuda.is_available() can still report True here while every real kernel "
                "launch fails -- this is the mismatch scripts/setup_env.ps1's "
                "-TorchCudaTag pin exists to avoid; reinstall a build matching the "
                "driver's ceiling."
            )

    gpu_name_smi = nvidia_smi_gpu_name()
    cuda_available = torch.cuda.is_available()
    print(f"  torch.cuda.is_available(): {cuda_available}")

    if not cuda_available:
        if build_kind == "cpu" and gpu_name_smi:
            problems.append(
                f"nvidia-smi sees a GPU ({gpu_name_smi}) but torch is the '+cpu' build, "
                "so cuda.is_available() is False. Reinstall from the CUDA wheel index "
                "(see scripts/setup_env.ps1 -IncludeCudaTorch) if GPU use is intended."
            )
        return

    device_name = torch.cuda.get_device_name(0)
    total_mem_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  device: {device_name}")
    print(f"  total VRAM: {total_mem_gib:.2f} GiB")

    arch_list = torch.cuda.get_arch_list()
    print(f"  torch.cuda.get_arch_list(): {arch_list}")
    if REQUIRED_ARCH in arch_list:
        print(f"  {REQUIRED_ARCH} present: yes")
    else:
        print(f"  {REQUIRED_ARCH} present: NO")
        problems.append(
            f"{REQUIRED_ARCH} is not in torch.cuda.get_arch_list() ({arch_list}). "
            "cuda.is_available() can still report True while this GPU's actual kernels "
            "are missing from the wheel -- ops will error or silently run on a mismatched "
            "arch. Install a torch build that still ships Pascal kernels."
        )

    try:
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(c).all().item())
        print(f"  GPU matmul (256x256): ran, finite results = {finite}")
        if not finite:
            problems.append("GPU matmul completed but produced non-finite values.")
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, this IS the check
        print(f"  GPU matmul (256x256): FAILED -- {exc}")
        problems.append(f"GPU matmul raised despite cuda.is_available()=True: {exc}")


def check_data() -> None:
    section("Dataset")
    data_root = REPO_ROOT / "data" / "odir5k"
    metadata_csv = data_root / "full_df.csv"

    if metadata_csv.exists():
        with open(metadata_csv, newline="", encoding="utf-8") as fh:
            n_rows = sum(1 for _ in csv.reader(fh)) - 1  # exclude header
        print(f"  {metadata_csv.relative_to(REPO_ROOT)}: found, {n_rows} rows")
    else:
        print(f"  {metadata_csv.relative_to(REPO_ROOT)}: not found")

    image_dir = data_root / "preprocessed_images"
    if image_dir.is_dir():
        n_files = sum(1 for p in image_dir.iterdir() if p.is_file())
        print(f"  {image_dir.relative_to(REPO_ROOT)}/: {n_files} files")
    else:
        print(f"  {image_dir.relative_to(REPO_ROOT)}/: not found")


def main() -> int:
    check_python()
    check_torch()
    check_data()

    section("Summary")
    if problems:
        print(f"  {len(problems)} problem(s) found:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("  no problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
