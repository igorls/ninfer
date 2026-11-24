"""Idle unload on a real server: the port keeps answering and the next request reloads the model."""

import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

IDLE_UNLOAD_SECONDS = 3
CONCURRENT_REQUESTS = 4


def get_json(endpoint: str, path: str, timeout: float = 5) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(endpoint + path, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def chat(endpoint: str, model: str, text: str, timeout: float = 600) -> dict:
    request = urllib.request.Request(
        endpoint + "/v1/chat/completions",
        data=json.dumps({
            "model": model, "stream": False, "max_tokens": 8,
            "messages": [{"role": "user", "content": text}],
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        assert response.status == 200, f"chat returned HTTP {response.status}"
        return json.load(response)


def wait_for(predicate, limit: float, what: str, process: subprocess.Popen):
    deadline = time.monotonic() + limit
    while True:
        if process.poll() is not None:
            raise AssertionError(f"server exited while waiting for {what}")
        value = predicate()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.1)


def health_model(endpoint: str) -> dict | None:
    try:
        status, body = get_json(endpoint, "/health", timeout=2)
    except OSError:
        return None
    return body if status == 200 else None


def main() -> int:
    artifact = os.environ.get("NINFER_TEST_ARTIFACT")
    if not artifact:
        print("SKIP: set NINFER_TEST_ARTIFACT to an explicit .ninfer artifact")
        return 77
    root = Path(__file__).resolve().parents[1] / ".agent-tmp/serve-idle-unload-test"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        log_path = Path(temporary) / "server.log"
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        endpoint = f"http://127.0.0.1:{port}"
        launch = {}
        if os.name == "nt":
            startup = subprocess.STARTUPINFO()
            startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            launch = {"creationflags": subprocess.CREATE_NEW_CONSOLE, "startupinfo": startup}
        extra = os.environ.get("NINFER_TEST_SERVE_ARGS", "").split()
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.argv[1], artifact, "--host", "127.0.0.1", "--port", str(port),
                 "--max-context", "8192", "--max-concurrency", str(CONCURRENT_REQUESTS),
                 "--spec", "none", "--host-cache-capacity", "0B", "--stats-interval", "1s",
                 "--idle-unload", f"{IDLE_UNLOAD_SECONDS}s", *extra],
                stdout=log, stderr=subprocess.STDOUT, **launch,
            )
            try:
                loaded = wait_for(lambda: (h := health_model(endpoint)) and
                                  h.get("model") == "loaded" and h, 300, "startup", process)
                assert loaded["status"] == "ok" and loaded["loads"] == 1, loaded
                assert loaded["idle_unload_ms"] == IDLE_UNLOAD_SECONDS * 1000, loaded
                status, models = get_json(endpoint, "/v1/models")
                assert status == 200, models
                model = models["data"][0]["id"]
                chat(endpoint, model, "Say hi.")

                # Idle past the limit: the model goes, the server stays.
                unloaded = wait_for(lambda: (h := health_model(endpoint)) and
                                    h.get("model") == "unloaded" and h,
                                    IDLE_UNLOAD_SECONDS + 60, "idle unload", process)
                assert unloaded["status"] == "ok" and unloaded["unloads"] == 1, unloaded
                status, listed = get_json(endpoint, "/v1/models")
                assert status == 200 and listed["data"][0]["id"] == model, listed
                status, props = get_json(endpoint, "/props")
                assert status == 200 and props["total_slots"] == CONCURRENT_REQUESTS, props
                status, residency = get_json(endpoint, "/admin/residency")
                assert status == 200 and residency["model"] == "unloaded", residency

                # Concurrent requests after the unload share one reload and are all served.
                started = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(CONCURRENT_REQUESTS) as pool:
                    answers = list(pool.map(
                        lambda i: chat(endpoint, model, f"Count to {i + 2}."),
                        range(CONCURRENT_REQUESTS)))
                reload_seconds = time.monotonic() - started
                assert all(a["choices"][0]["message"] is not None for a in answers), answers
                status, after = get_json(endpoint, "/health")
                assert status == 200 and after["model"] == "loaded", after
                assert after["loads"] == 2, f"expected one shared reload, health says {after}"

                # Disabling the limit at run time keeps the reloaded model resident.
                request = urllib.request.Request(
                    endpoint + "/admin/residency", data=b'{"idle_unload_ms": 0}',
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=5) as response:
                    assert json.load(response)["idle_unload_ms"] == 0
                time.sleep(IDLE_UNLOAD_SECONDS + 2)
                status, kept = get_json(endpoint, "/health")
                assert kept["model"] == "loaded" and kept["unloads"] == 1, kept
            except BaseException:
                print(log_path.read_text(errors="replace"), file=sys.stderr)
                raise
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        text = log_path.read_text(errors="replace")
        assert text.count("model unloaded") == 1, "unload was not logged once"
        assert text.count("model reloading") == 1, "concurrent requests started more than one reload"
        assert text.count("model reloaded") == 1, "reload completion was not logged"
        print(f"PASS idle unload kept the port up; {CONCURRENT_REQUESTS} concurrent requests "
              f"shared one reload and finished in {reload_seconds:.1f}s "
              f"(engine load {after['last_load_seconds']:.1f}s, unload "
              f"{after['last_unload_seconds']:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
