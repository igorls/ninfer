"""Colab VM setup for 27B reasoning-router collection (run with `colab exec -f`).

Extracts /content/source.tar.gz (a `git archive --prefix=ninfer/` of the collection commit),
records the environment, then starts two background jobs and returns immediately:
the Linux build of ninfer-reasoning-collect and the pinned artifact download + SHA-256 check.
Poll build.log for BUILD_COMPLETE and download.log for VERIFIED.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

R = Path("/content/rr")
REPO = "neroued/Qwen3.8-27B-nvfp4-NInfer"
# Revision 11dbbbbb holds the container-v2 file byte-identical to the local artifact the
# earlier study used; HF main is a container-v3 file this branch cannot load.
REVISION = "11dbbbbbc33db198afe2f02c9232c771ff7031be"
FILENAME = "qwen3_8_27b_nvfp4.ninfer"
SIZE = 23_719_496_192
SHA256 = "552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c"

R.mkdir(exist_ok=True)
for name in ("logs", "export", "inputs"):
    (R / name).mkdir(exist_ok=True)
env = {}
for key, command in (("gpu", ["nvidia-smi"]), ("nvcc", ["nvcc", "--version"]),
                     ("cmake", ["cmake", "--version"]), ("gxx", ["g++", "--version"]),
                     ("memory", ["free", "-h"]), ("disk", ["df", "-h", "/content"]),
                     ("cpus", ["nproc"]), ("python", [sys.executable, "--version"])):
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=60)
        env[key] = done.stdout + done.stderr
    except Exception as error:  # noqa: BLE001 - recorded, not fatal
        env[key] = f"unavailable: {error}"
(R / "environment.json").write_text(json.dumps(env, indent=2))
print(env["gpu"][:1200])

for path in Path("/content").glob("*.jsonl"):
    shutil.move(str(path), str(R / "inputs" / path.name))
for path in Path("/content").glob("*.json"):
    shutil.move(str(path), str(R / "inputs" / path.name))
if not (R / "ninfer").exists():
    with tarfile.open("/content/source.tar.gz") as archive:
        archive.extractall(R, filter="data")
print("source:", (R / "ninfer" / "CMakeLists.txt").exists())

# NInfer needs CUDA 13.1+. Colab images differ: some ship 13.3, others only 12.8 with a CUDA 13.0
# driver (580.x), which runs 13.x-built binaries through minor-version compatibility.
(R / "build.sh").write_text(f"""set -e
trap 'echo BUILD_FAILED' ERR
NVCC=$(ls -d /usr/local/cuda-13.[1-9]*/bin/nvcc 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$NVCC" ]; then
  echo "installing cuda-toolkit-13-3"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-13-3
  NVCC=/usr/local/cuda-13.3/bin/nvcc
fi
echo "nvcc: $NVCC"; "$NVCC" --version | tail -2
rm -rf {R}/build
cmake -S {R}/ninfer -B {R}/build -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_MEDIA=OFF -DBUILD_TESTING=OFF \\
  -DCMAKE_CUDA_COMPILER="$NVCC" -DCUDAToolkit_ROOT="$(dirname "$(dirname "$NVCC")")"
cmake --build {R}/build --target ninfer-reasoning-collect -j
echo BUILD_COMPLETE
""")
(R / "download.py").write_text(f"""import hashlib, json, os
from huggingface_hub import hf_hub_download
path = hf_hub_download({REPO!r}, {FILENAME!r}, revision={REVISION!r},
                       local_dir="/content/models/qwen27b", token=False)
size = os.path.getsize(path)
assert size == {SIZE}, ("size", size)
with open(path, "rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
assert digest == {SHA256!r}, ("sha256", digest)
json.dump({{"path": path, "repo": {REPO!r}, "revision": {REVISION!r}, "size": size, "sha256": digest}},
          open({str(R / 'artifact.json')!r}, "w"), indent=2)
print("VERIFIED", digest, flush=True)
""")
def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def marker(name):
    path = R / name
    return path.read_text(errors="replace") if path.exists() else ""


if "BUILD_COMPLETE" in marker("build.log"):
    print("build: already complete")
elif running(str(R / "build.sh")):
    print("build: already running")
else:
    subprocess.Popen(["bash", str(R / "build.sh")], stdout=open(R / "build.log", "w"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    print("build: started")
if "VERIFIED" in marker("download.log"):
    print("download: already verified")
elif running(str(R / "download.py")):
    print("download: already running")
else:
    subprocess.Popen([sys.executable, str(R / "download.py")], stdout=open(R / "download.log", "w"),
                     stderr=subprocess.STDOUT, start_new_session=True,
                     env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    print("download: started")
