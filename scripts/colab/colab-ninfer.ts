#!/usr/bin/env bun
/**
 * Ephemeral Google Colab G4 (RTX PRO 6000 Blackwell, sm_120a) launcher,
 * precompiled-binary packager, and private SSH/Tailscale bridge for NInfer.
 *
 * Keeps the Colab VM completely off Tailscale and off the public internet:
 *   - `tunnel`: Colab (127.0.0.1:8010) <-- colab ssh --proxy-mode --> Workstation:<port>
 *   - `bridge`: Workstation:<port> -- ssh -R (outbound, zero firewall edits) --> Remote Host 127.0.0.1:<port>
 *
 * Usage:
 *   bun scripts/colab/colab-ninfer.ts up [--session ninfer-g4] [--gpu G4] [--build]
 *                                        [--model-repo ...] [--model-file ...] [--model-id ...]
 *                                        [--context-capacity auto] [--max-concurrency 8] [--max-context 65536]
 *   bun scripts/colab/colab-ninfer.ts tunnel [--session ninfer-g4] [--port 18010] [--bind 127.0.0.1|auto]
 *   bun scripts/colab/colab-ninfer.ts bridge --target root@<host> [--session ninfer-g4] [--port 18010]
 *   bun scripts/colab/colab-ninfer.ts package [--session ninfer-g4] [--workdir /content/ninfer-colab]
 *                                             [--out ninfer-serve-linux-x86_64-sm120a.tar.gz]
 *                                             [--publish-release colab-linux-sm120a]
 *   bun scripts/colab/colab-ninfer.ts status [--session ninfer-g4] [--port 18010]
 *   bun scripts/colab/colab-ninfer.ts stop-server [--session ninfer-g4] [--workdir /content/ninfer-colab]
 *   bun scripts/colab/colab-ninfer.ts down [--session ninfer-g4]
 */

import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SETUP_SCRIPT = join(HERE, "setup-g4.sh");
const DEFAULT_SESSION = "ninfer-g4";
const DEFAULT_ASSET = "ninfer-serve-linux-x86_64-sm120a.tar.gz";

function argValue(args: readonly string[], flag: string, fallback: string): string {
  const idx = args.indexOf(flag);
  if (idx >= 0 && args[idx + 1] && !args[idx + 1]!.startsWith("--")) return args[idx + 1]!;
  const prefix = `${flag}=`;
  const hit = args.find((a) => a.startsWith(prefix));
  return hit ? hit.slice(prefix.length) : fallback;
}

function hasFlag(args: readonly string[], flag: string): boolean {
  return args.includes(flag);
}

function shellQuote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

function runSync(cmd: readonly string[], label: string): void {
  console.log(`[colab-ninfer] ${label}: ${cmd.join(" ")}`);
  const proc = Bun.spawnSync([...cmd], { stdout: "inherit", stderr: "inherit" });
  if (proc.exitCode !== 0) {
    throw new Error(`${label} failed with exit code ${proc.exitCode}`);
  }
}

function capture(cmd: readonly string[]): { ok: boolean; stdout: string; stderr: string } {
  const proc = Bun.spawnSync([...cmd], { stdout: "pipe", stderr: "pipe" });
  return {
    ok: proc.exitCode === 0,
    stdout: new TextDecoder().decode(proc.stdout).trim(),
    stderr: new TextDecoder().decode(proc.stderr).trim(),
  };
}

function resolveBindHost(bindArg: string): string {
  if (bindArg !== "auto") return bindArg;
  const ts = capture(["tailscale", "ip", "-4"]);
  const ip = ts.stdout.split(/\r?\n/)[0]?.trim() ?? "";
  if (ts.ok && /^\d+\.\d+\.\d+\.\d+$/.test(ip)) return ip;
  return "127.0.0.1";
}

function sessionExists(session: string): boolean {
  const res = capture(["colab", "sessions"]);
  return res.ok && res.stdout.includes(`[${session}]`);
}

function colabSshTunnelArgs(session: string, bindHost: string, localPort: string, remotePort: string): string[] {
  const colabBin = process.platform === "win32" ? "colab.exe" : "colab";
  const proxyCmd = `${colabBin} ssh --proxy-mode -s ${session}`;
  return [
    "ssh",
    "-N",
    "-T",
    "-o",
    "ExitOnForwardFailure=yes",
    "-o",
    "ServerAliveInterval=20",
    "-o",
    "ServerAliveCountMax=3",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    `ProxyCommand=${proxyCmd}`,
    "-L",
    `${bindHost}:${localPort}:127.0.0.1:${remotePort}`,
    ...(bindHost !== "127.0.0.1" ? ["-L", `127.0.0.1:${localPort}:127.0.0.1:${remotePort}`] : []),
    "root@colab",
  ];
}

async function cmdUp(args: readonly string[]): Promise<void> {
  const session = argValue(args, "--session", DEFAULT_SESSION);
  const gpu = argValue(args, "--gpu", "G4");
  if (!existsSync(SETUP_SCRIPT)) {
    throw new Error(`Missing setup script: ${SETUP_SCRIPT}`);
  }

  if (!sessionExists(session)) {
    runSync(["colab", "new", "--session", session, "--gpu", gpu], `Allocating Colab ${gpu} session ${session}`);
  } else {
    console.log(`[colab-ninfer] Reusing existing Colab session '${session}'`);
  }

  const envMap: Record<string, string> = {};
  const setOpt = (flag: string, envName: string) => {
    const val = argValue(args, flag, "");
    if (val) envMap[envName] = val;
  };
  if (hasFlag(args, "--build")) envMap["NINFER_BUILD_FROM_SOURCE"] = "1";
  setOpt("--workdir", "NINFER_WORKDIR");
  setOpt("--release-tag", "NINFER_RELEASE_TAG");
  setOpt("--binary-url", "NINFER_BINARY_URL");
  setOpt("--branch", "NINFER_BRANCH");
  setOpt("--model-repo", "NINFER_MODEL_REPO");
  setOpt("--model-file", "NINFER_MODEL_FILE");
  setOpt("--model-id", "NINFER_MODEL_ID");
  setOpt("--remote-port", "NINFER_PORT");
  setOpt("--max-context", "NINFER_MAX_CONTEXT");
  setOpt("--context-capacity", "NINFER_CONTEXT_CAPACITY");
  setOpt("--max-concurrency", "NINFER_MAX_CONCURRENCY");
  setOpt("--kv-dtype", "NINFER_KV_DTYPE");

  const envPrefix = Object.entries(envMap)
    .map(([k, v]) => `${k}=${shellQuote(v)}`)
    .join(" ");

  runSync(["colab", "upload", "-s", session, SETUP_SCRIPT, "/content/setup-ninfer-g4.sh"], "Uploading setup-g4.sh");
  runSync(
    [
      "colab",
      "ssh",
      "-s",
      session,
      "--",
      "bash",
      "-lc",
      `sed -i 's/\\r$//' /content/setup-ninfer-g4.sh && chmod +x /content/setup-ninfer-g4.sh && ${envPrefix ? `${envPrefix} ` : ""}/content/setup-ninfer-g4.sh`,
    ],
    "Bootstrapping ninfer-serve on Colab G4",
  );

  console.log(`[colab-ninfer] Ready on Colab session '${session}' (127.0.0.1:${envMap["NINFER_PORT"] ?? "8010"}).`);
  console.log(`[colab-ninfer] Local tunnel:  bun scripts/colab/colab-ninfer.ts tunnel --session ${session}`);
  console.log(
    `[colab-ninfer] Remote bridge: bun scripts/colab/colab-ninfer.ts bridge --session ${session} --target root@<host>`,
  );
}

async function cmdTunnel(args: readonly string[]): Promise<void> {
  const session = argValue(args, "--session", DEFAULT_SESSION);
  const localPort = argValue(args, "--port", "18010");
  const remotePort = argValue(args, "--remote-port", "8010");
  const bindHost = resolveBindHost(argValue(args, "--bind", "127.0.0.1"));

  console.log(`[colab-ninfer] Opening private SSH tunnel: ${bindHost}:${localPort} -> [${session}] 127.0.0.1:${remotePort}`);
  console.log(`[colab-ninfer] Endpoint: http://${bindHost}:${localPort}/v1`);
  console.log(`[colab-ninfer] Press Ctrl+C to close (session keeps running until 'stop-server' or 'down').`);

  const proc = Bun.spawn(colabSshTunnelArgs(session, bindHost, localPort, remotePort), {
    stdout: "inherit",
    stderr: "inherit",
  });
  process.exit(await proc.exited);
}

async function cmdBridge(args: readonly string[]): Promise<void> {
  const session = argValue(args, "--session", DEFAULT_SESSION);
  const target = argValue(args, "--target", "");
  const port = argValue(args, "--port", "18010");
  const remotePort = argValue(args, "--remote-port", "8010");
  if (!target) {
    throw new Error("bridge requires --target <user@host> (e.g. --target root@bentokit-tribuno-yduqs)");
  }

  console.log(
    `[colab-ninfer] Two-hop zero-firewall bridge: [${target}] 127.0.0.1:${port} -> localhost:${port} -> [${session}] 127.0.0.1:${remotePort}`,
  );
  const hop1 = Bun.spawn(colabSshTunnelArgs(session, "127.0.0.1", port, remotePort), {
    stdout: "inherit",
    stderr: "inherit",
  });

  // Wait briefly for hop 1 to listen on localhost:<port>.
  const deadline = Date.now() + 25_000;
  let ready = false;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/v1/models`, { signal: AbortSignal.timeout(1500) });
      if (res.ok) {
        ready = true;
        break;
      }
    } catch {
      // retry until tunnel comes up
    }
    await Bun.sleep(500);
  }
  if (!ready) {
    hop1.kill();
    throw new Error(`Colab SSH tunnel on 127.0.0.1:${port} did not become ready within 25s`);
  }

  console.log(`[colab-ninfer] Hop 1 ready (http://127.0.0.1:${port}/v1). Opening reverse forward to ${target}...`);
  const hop2 = Bun.spawn(
    [
      "ssh",
      "-N",
      "-T",
      "-o",
      "BatchMode=yes",
      "-o",
      "ExitOnForwardFailure=yes",
      "-o",
      "ServerAliveInterval=20",
      "-o",
      "ServerAliveCountMax=3",
      "-o",
      "StrictHostKeyChecking=no",
      "-R",
      `127.0.0.1:${port}:127.0.0.1:${port}`,
      target,
    ],
    { stdout: "inherit", stderr: "inherit" },
  );

  const shutdown = () => {
    hop2.kill();
    hop1.kill();
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  console.log(`[colab-ninfer] Bridge LIVE on ${target} at http://127.0.0.1:${port}/v1`);
  const code = await Promise.race([hop1.exited, hop2.exited]);
  shutdown();
  process.exit(code);
}

async function cmdPackage(args: readonly string[]): Promise<void> {
  const session = argValue(args, "--session", DEFAULT_SESSION);
  const workdir = argValue(args, "--workdir", "/content/ninfer-colab");
  const outPath = resolve(argValue(args, "--out", DEFAULT_ASSET));
  const releaseTag = argValue(args, "--publish-release", "");
  const repo = argValue(args, "--repo", "igorls/ninfer");

  // Package ninfer-serve + CUDA runtime libs required on fresh Colab images + build-info.json.
  const remoteScript = [
    "set -euo pipefail",
    `W=${shellQuote(workdir)}`,
    'BIN="$W/build/apps/ninfer-serve"',
    'test -x "$BIN" || { echo "Missing compiled binary: $BIN" >&2; exit 1; }',
    'STAGE="$W/pkg-stage"',
    'rm -rf "$STAGE" && mkdir -p "$STAGE/lib"',
    'cp -L "$BIN" "$STAGE/ninfer-serve"',
    'chmod +x "$STAGE/ninfer-serve"',
    // Bundle CUDA 13.3 runtime shared libs referenced by ninfer-serve so fresh Colab VMs need zero cuda-toolkit apt install.
    'ldd "$BIN" | awk \'/\\/usr\\/local\\/cuda/ {print $3}\' | while read -r lib; do [ -f "$lib" ] && cp -L "$lib" "$STAGE/lib/"; done',
    'COMMIT=$(git -C "$W/ninfer" rev-parse HEAD 2>/dev/null || echo unknown)',
    'SHORT=$(git -C "$W/ninfer" rev-parse --short HEAD 2>/dev/null || echo unknown)',
    'NVCC=$(/usr/local/cuda-13.3/bin/nvcc --version 2>/dev/null | tail -n 1 || echo unknown)',
    'GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || echo unknown)',
    'OS=$(. /etc/os-release && echo "${PRETTY_NAME:-linux}")',
    'cat > "$STAGE/build-info.json" <<EOF',
    "{",
    '  "commit": "$COMMIT",',
    '  "short_commit": "$SHORT",',
    '  "target": "rtx5090 (sm_120a)",',
    '  "nvcc": "$NVCC",',
    '  "gpu": "$GPU",',
    '  "os": "$OS",',
    '  "built_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"',
    "}",
    "EOF",
    'tar -czf "$W/ninfer-serve-linux-x86_64-sm120a.tar.gz" -C "$STAGE" .',
    'ls -lh "$W/ninfer-serve-linux-x86_64-sm120a.tar.gz"',
    'sha256sum "$W/ninfer-serve-linux-x86_64-sm120a.tar.gz"',
  ].join("\n");

  runSync(
    ["colab", "ssh", "-s", session, "--", "bash", "-lc", remoteScript],
    `Packaging precompiled ninfer-serve from Colab session '${session}'`,
  );
  runSync(
    ["colab", "download", "-s", session, `${workdir}/${DEFAULT_ASSET}`, outPath],
    `Downloading ${DEFAULT_ASSET} to ${outPath}`,
  );

  if (releaseTag) {
    const view = capture(["gh", "release", "view", releaseTag, "-R", repo]);
    if (!view.ok) {
      runSync(
        [
          "gh",
          "release",
          "create",
          releaseTag,
          outPath,
          "-R",
          repo,
          "--title",
          `NInfer Linux sm_120a Prebuilt (${releaseTag})`,
          "--notes",
          "Precompiled Linux x86_64 Blackwell (sm_120a) ninfer-serve bundle for fast Colab G4 bootstrapping.",
        ],
        `Creating GitHub release ${releaseTag} on ${repo}`,
      );
    } else {
      runSync(
        ["gh", "release", "upload", releaseTag, outPath, "-R", repo, "--clobber"],
        `Uploading ${DEFAULT_ASSET} to release ${releaseTag} on ${repo}`,
      );
    }
  }
}

async function cmdStatus(args: readonly string[]): Promise<void> {
  const localPort = argValue(args, "--port", "18010");
  const bindHost = resolveBindHost(argValue(args, "--bind", "127.0.0.1"));
  runSync(["colab", "sessions"], "Active Colab sessions");
  runSync(["colab", "usage"], "Colab compute units");
  for (const host of [...new Set(["127.0.0.1", bindHost])]) {
    const url = `http://${host}:${localPort}/v1/models`;
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(3000) });
      console.log(`[colab-ninfer] ${url} -> HTTP ${res.status} ${(await res.text()).slice(0, 200)}`);
    } catch (error) {
      console.log(`[colab-ninfer] ${url} -> unreachable (${error instanceof Error ? error.message : String(error)})`);
    }
  }
}

async function cmdStopServer(args: readonly string[]): Promise<void> {
  const session = argValue(args, "--session", DEFAULT_SESSION);
  const workdir = argValue(args, "--workdir", "/content/ninfer-colab");
  runSync(
    [
      "colab",
      "ssh",
      "-s",
      session,
      "--",
      "bash",
      "-lc",
      `if [ -f ${shellQuote(`${workdir}/ninfer-serve.pid`)} ]; then kill "$(cat ${shellQuote(`${workdir}/ninfer-serve.pid`)})" 2>/dev/null || true; rm -f ${shellQuote(`${workdir}/ninfer-serve.pid`)}; echo "Stopped ninfer-serve (session kept alive)"; else echo "No ninfer-serve.pid in ${workdir}"; fi`,
    ],
    `Stopping ninfer-serve on '${session}' (leaving session & other workloads untouched)`,
  );
}

async function cmdDown(args: readonly string[]): Promise<void> {
  const session = argValue(args, "--session", DEFAULT_SESSION);
  runSync(["colab", "stop", "-s", session], `Stopping Colab session ${session}`);
}

const [subcommand, ...rest] = process.argv.slice(2);
switch (subcommand) {
  case "up":
    await cmdUp(rest);
    break;
  case "tunnel":
    await cmdTunnel(rest);
    break;
  case "bridge":
    await cmdBridge(rest);
    break;
  case "package":
    await cmdPackage(rest);
    break;
  case "status":
    await cmdStatus(rest);
    break;
  case "stop-server":
    await cmdStopServer(rest);
    break;
  case "down":
    await cmdDown(rest);
    break;
  default:
    console.log(
      "Usage: bun scripts/colab/colab-ninfer.ts <up|tunnel|bridge|package|status|stop-server|down> [options]",
    );
    process.exit(subcommand ? 1 : 0);
}
