
from pathlib import Path
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys

# ============================================================
# SCRIPT 00: DEEP LEARNING ENVIRONMENT CHECK
#
# Checks:
# - Python and system information
# - Existing virtual environment
# - CPU, RAM, disk space
# - Deep learning package versions
# - PyTorch GPU availability
# - NVIDIA GPU information (when available)
#
# No dataset loading.
# No model downloads.
# No model training.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = REPORTS / "00_environment_report.json"
OUTPUT_TXT = REPORTS / "00_environment_report.txt"

PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "tokenizers",
    "safetensors",
    "sentencepiece",
    "evaluate",
    "matplotlib",
]

def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"

def memory_gb():
    try:
        import psutil
        m = psutil.virtual_memory()
        return {
            "total_gb": round(m.total / 1024**3, 2),
            "available_gb": round(m.available / 1024**3, 2),
        }
    except ImportError:
        pass

    try:
        values = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0])

        return {
            "total_gb": round(
                values["MemTotal"] / 1024**2, 2
            ),
            "available_gb": round(
                values["MemAvailable"] / 1024**2, 2
            ),
        }
    except (OSError, KeyError, ValueError):
        return {"status": "UNAVAILABLE"}

def check_pytorch():
    information = {}

    try:
        torch = importlib.import_module("torch")
        information["version"] = torch.__version__
        information["cuda_available"] = (
            torch.cuda.is_available()
        )
        information["torch_cuda_version"] = (
            torch.version.cuda
        )

        if torch.cuda.is_available():
            information["device"] = "cuda"
            information["gpu_count"] = (
                torch.cuda.device_count()
            )

            gpus = []

            for index in range(
                torch.cuda.device_count()
            ):
                properties = (
                    torch.cuda.get_device_properties(index)
                )

                gpus.append({
                    "index": index,
                    "name": properties.name,
                    "total_memory_gb": round(
                        properties.total_memory / 1024**3,
                        2
                    ),
                })

            information["gpus"] = gpus

            # Basic allocation test.
            try:
                tensor = torch.tensor(
                    [1.0, 2.0, 3.0],
                    device="cuda"
                )
                information["cuda_tensor_test"] = (
                    float(tensor.sum().item()) == 6.0
                )
                del tensor
            except Exception as error:
                information["cuda_tensor_test"] = False
                information["cuda_error"] = str(error)

        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            information["device"] = "mps"
        else:
            information["device"] = "cpu"

    except Exception as error:
        information["status"] = "ERROR"
        information["error"] = repr(error)

    return information

def check_nvidia_smi():
    if shutil.which("nvidia-smi") is None:
        return "NVIDIA-SMI NOT AVAILABLE"

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,"
                "memory.free,driver_version",
                "--format=csv,noheader"
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False
        )

        if result.returncode != 0:
            return result.stderr.strip()

        return result.stdout.strip()

    except Exception as error:
        return repr(error)

print("=" * 76)
print("SCRIPT 00: DEEP LEARNING ENVIRONMENT CHECK")
print("=" * 76)

system = {
    "python_version": sys.version.split()[0],
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "cpu_threads": os.cpu_count(),
    "virtual_environment": os.environ.get(
        "VIRTUAL_ENV",
        "NOT DETECTED"
    ),
    "memory": memory_gb(),
    "disk_free_gb": round(
        shutil.disk_usage(ROOT).free / 1024**3,
        2
    ),
}

print("\nSYSTEM INFORMATION")
print("Python:", system["python_version"])
print("Python executable:", system["python_executable"])
print("Platform:", system["platform"])
print("CPU threads:", system["cpu_threads"])
print(
    "Virtual environment:",
    system["virtual_environment"]
)
print("RAM:", system["memory"])
print(
    "Free disk space (GB):",
    system["disk_free_gb"]
)

packages = {
    name: package_version(name)
    for name in PACKAGE_NAMES
}

print("\nPACKAGE AVAILABILITY")

for name, version in packages.items():
    print(f"{name}: {version}")

pytorch = check_pytorch()

print("\nPYTORCH HARDWARE")

for key, value in pytorch.items():
    print(f"{key}: {value}")

nvidia = check_nvidia_smi()

print("\nNVIDIA DRIVER INFORMATION")
print(nvidia)

# ------------------------------------------------------------
# Research readiness assessment
# ------------------------------------------------------------

torch_ok = (
    packages["torch"] != "NOT INSTALLED"
    and "error" not in pytorch
)

transformers_ok = (
    packages["transformers"] != "NOT INSTALLED"
)

datasets_ok = (
    packages["datasets"] != "NOT INSTALLED"
)

accelerate_ok = (
    packages["accelerate"] != "NOT INSTALLED"
)

gpu_available = pytorch.get("device") in (
    "cuda", "mps"
)

print("\nRESEARCH READINESS")

print("PyTorch installed:", torch_ok)
print("Transformers installed:", transformers_ok)
print("Datasets installed:", datasets_ok)
print("Accelerate installed:", accelerate_ok)
print("GPU backend detected:", gpu_available)

if torch_ok:
    print("Basic PyTorch availability: PASS")
else:
    print("Basic PyTorch availability: NEEDS ATTENTION")

if all([
    torch_ok,
    transformers_ok,
    datasets_ok,
    accelerate_ok
]):
    readiness = "CORE_PACKAGES_PRESENT"
else:
    readiness = "MISSING_CORE_PACKAGES"

print("Environment status:", readiness)
print("Transformer runtime compatibility: NOT YET VERIFIED")
print("Dataset integrity check: PENDING")
print("Model training: NOT STARTED")
print("Test evaluation: NOT STARTED")

report = {
    "script": "00_environment_check",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "system": system,
    "packages": packages,
    "pytorch": pytorch,
    "nvidia_smi": nvidia,
    "readiness": readiness,
    "transformer_runtime_verified": False,
    "dataset_loaded": False,
    "model_training_started": False,
    "test_evaluation_started": False,
}

OUTPUT_JSON.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

lines = [
    "DEEP LEARNING ENVIRONMENT REPORT",
    "=" * 76,
    "",
    "SYSTEM INFORMATION",
    json.dumps(system, indent=2),
    "",
    "PACKAGE AVAILABILITY",
    *[
        f"{name}: {version}"
        for name, version in packages.items()
    ],
    "",
    "PYTORCH HARDWARE",
    json.dumps(pytorch, indent=2),
    "",
    "NVIDIA DRIVER INFORMATION",
    str(nvidia),
    "",
    "RESEARCH READINESS",
    f"Environment status: {readiness}",
    "Transformer runtime compatibility: NOT YET VERIFIED",
    "Dataset integrity check: PENDING",
    "Model training: NOT STARTED",
    "Test evaluation: NOT STARTED",
]

OUTPUT_TXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

print("\nFILES GENERATED")
print(OUTPUT_JSON)
print(OUTPUT_TXT)

print("\nScript 00 completed.")
