"""Parse the JavaScript shipped in the Supervisor's executable HTML resource, and check the
capacity arithmetic the Settings page shows against a plan the engine reported."""

import argparse
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.scripts = []
        self.current = None
        self.module = False
        self.name = ""

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            attrs = dict(attrs)
            self.current = [] if attrs.get("type", "") not in ("application/json", "importmap") else None
            self.module = attrs.get("type", "") == "module"
            self.name = attrs.get("id") or ""

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.current is not None:
            self.scripts.append(("".join(self.current), self.module, self.name))
            self.current = None


GIB = 1 << 30

# A served 27B at fp8 KV with DFlash2, 8 lanes, a 20 GiB pool and a 6 GiB reserve, as /admin/vram
# and the request log's server_start reported it. The engine logged a token estimate of 607950.
STATE = {
    "engine": {"state": "Running"},
    "health": {"status": 200},
    "device_memory": {"ok": True, "total_bytes": 102641958912, "used_bytes": 85738913792},
    "admin_vram": {
        "device": {"total_bytes": 102641958912, "free_bytes": 18774290432, "used_bytes": 83867668480},
        "plan": {
            "runtime_reservation_bytes": 22855900416,
            "minimum_context_capacity_bytes": 10774353920,
            "context_capacity_bytes": 21474836480,
            "context_capacity_automatic": False,
            "headroom_bytes": GIB,
            "available_after_weights_bytes": 37069344768,
            "cuda_graph_rebuild_estimate_bytes": 0,
            "effective_concurrency": 8,
        },
        "resources": {"allocations": {"device_bytes": 46111074560, "device_weights_bytes": 23255174144}},
        "context": {"capacities": [
            {"domain": "device", "capacity_bytes": 21474836480, "block_bytes": 195897344},
            {"domain": "host", "capacity_bytes": 10157113344, "block_bytes": 195897344},
        ]},
        "desktop_reserve": {"configured_bytes": 6 * GIB},
    },
    "engine_start": {
        "token_estimate": 607950, "device_capacity_bytes": 21474836480, "block_bytes": 195897344,
        "max_context": 262144, "max_concurrency": 8, "prefill_chunk": 2048, "kv_cache": "fp8",
        "speculative_backend": "dflash2", "draft_tokens": 7, "prefix_reuse": True, "vision": True,
    },
}

CHECKS = r"""
const assert = require('assert');
const C = window.NInferCapacity, GiB = 1073741824;
const state = JSON.parse(process.env.STATE);
const draft = extra => Object.assign({auto: false, capacity: 20 * GiB, reserve: 6 * GiB, maxContext: 262144,
  concurrency: 8, prefillChunk: 2048, draftTokens: 7, prefixReuse: true, changed: [], formatChanged: false}, extra);

assert.strictEqual(C.bytesOf('20GiB'), 20 * GiB);
assert.strictEqual(C.bytesOf('21474836480'), 20 * GiB, 'a bare integer is bytes, as the engine reads it');
assert.strictEqual(C.bytesOf('0B'), null);
assert.strictEqual(C.bytesOf('20.1GiB'), null, 'a fraction must land on a whole byte');
assert.strictEqual(C.normalize('20'), '20GiB', 'a bare number typed in the GiB field is GiB');
assert.strictEqual(C.normalize('20.5 gib'), '20.5GiB');
assert.strictEqual(C.normalize('20.1GiB'), '20582MiB');
assert.strictEqual(C.normalize('lots'), null);

const base = C.baseline(state);
assert.ok(base && base.shape, 'a ready engine with a matching start record gives a plan and its token shape');
assert.strictEqual(C.baseline({...state, health: {status: 503}}), null, 'no plan without a ready engine');
const stale = C.baseline({...state, engine_start: {...state.engine_start, device_capacity_bytes: 1}});
assert.ok(stale && !stale.shape, 'a start record for another pool gives no token figures');

const p = C.project(base, draft());
assert.strictEqual(p.tokens, 607950, 'the running pool reproduces the engine estimate');
assert.strictEqual(p.fullLength, 2);
assert.ok(p.perRequest > 67000 && p.perRequest < 68500, 'eight lanes share what their state leaves: ' + p.perRequest);
assert.strictEqual(p.minimum, 10774353920, 'unchanged shape reports the engine minimum');
assert.ok(p.minimumExact && p.fits && p.enough);
assert.strictEqual(p.reserveMaxGiB, 12, 'the reserve limit matches /api/desktop-reserve');
assert.strictEqual(p.left, 16903045120, 'what stays free is the live reading with the same pool');

const modelled = C.project(base, draft({changed: ['test']}));
assert.strictEqual(modelled.minimum, 10774353920, 'the working-set model reproduces the engine minimum');
assert.ok(!modelled.minimumExact);
const shorter = C.project(base, draft({maxContext: 65536, changed: ['max context']}));
assert.strictEqual(shorter.minimum, 22 * 195897344, 'at 64K the checkpoint handover set (8 + 2 states, 12 KV blocks) sets the minimum');
assert.strictEqual(shorter.fullLength, 7, 'and 109 blocks hold seven 64K requests');

const auto = C.project(base, draft({auto: true, capacity: null}));
assert.strictEqual(auto.pool, 157 * 195897344, 'auto takes whole blocks of what the reserve leaves');
assert.ok(auto.fits && auto.room - auto.pool < 195897344);
assert.strictEqual(C.project(base, draft({capacity: 40 * GiB})).fits, false, 'a pool beyond free memory does not fit');
assert.strictEqual(C.project(base, draft({capacity: 8 * GiB})).enough, false, 'a pool below the working sets is refused');
const bigger = C.project(base, draft({capacity: 12 * GiB, reserve: 16 * GiB}));
assert.ok(bigger.fits && bigger.reserveMaxGiB >= 16, 'a smaller pool makes room for a larger reserve');
const other = C.project(base, draft({formatChanged: true, changed: ['KV cache format']}));
assert.strictEqual(other.tokens, null, 'token figures wait for a restart when the KV format changes');
console.log('capacity arithmetic ok');
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True)
    parser.add_argument("--html", required=True, type=Path)
    args = parser.parse_args()
    html = Scripts()
    html.feed(args.html.read_text(encoding="utf-8"))
    if not html.scripts:
        raise AssertionError("Supervisor resource contains no JavaScript")
    for index, (script, module, _) in enumerate(html.scripts):
        command = [args.node, "--check", "-"]
        if module:
            command.insert(1, "--input-type=module")
        result = subprocess.run(command, input=script, text=True,
                                encoding="utf-8", capture_output=True, check=False)
        if result.returncode:
            raise AssertionError(f"Supervisor script {index} cannot parse:\n{result.stderr}")
    print(f"Parsed {len(html.scripts)} Supervisor scripts")

    model = next((script for script, _, name in html.scripts if name == "capacity-model"), None)
    if model is None:
        raise AssertionError("Supervisor resource has no capacity-model script")
    result = subprocess.run([args.node, "-"], input="const window = {};\n" + model + CHECKS, text=True,
                            encoding="utf-8", capture_output=True, check=False,
                            env={**os.environ, "STATE": json.dumps(STATE)})
    if result.returncode:
        raise AssertionError(f"Capacity arithmetic failed:\n{result.stderr}")
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
