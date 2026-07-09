"""Two-way sync between the training daemon and the GitHub repo.

Inbound  (app -> daemon): the app commits polytopia-rl/control/control.json
  (rule changes) and polytopia-rl/control/requests/<id>.json (game requests)
  to the working branch; we poll with plain git fetch/show.
Outbound (daemon -> app): data.json, game replays, and an index are committed
  to the gh-pages branch, which GitHub Pages serves to the app.

No API tokens are needed on this side — the container's git remote already
has push access to both branches.
"""

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
REPO = "/home/user/AutoGPT"
BRANCH = "claude/polytopia-rl-setup-ayk6ia"
GHP = ROOT / "ghp"                             # gh-pages worktree
CONTROL_REF = f"origin/{BRANCH}:polytopia-rl/control/control.json"
REQ_DIR_REF = f"origin/{BRANCH}:polytopia-rl/control/requests"
STATE = RUNS / "sync_state.json"
COMMIT_TRAILER = ("\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n"
                  "Claude-Session: https://claude.ai/code/session_01NjHHoEZKt36fM8gyubySY6")


def _git(*args, cwd=REPO, check=False):
    r = subprocess.run(["git", "-C", str(cwd), *args],
                       capture_output=True, text=True, timeout=120)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r


def _state():
    try:
        return json.loads(STATE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"control_hash": None, "processed": []}


def _save_state(s):
    STATE.write_text(json.dumps(s))


def ensure_ghp():
    if (GHP / ".git").exists():
        return
    _git("worktree", "prune")
    _git("fetch", "-q", "origin", "gh-pages")
    _git("worktree", "add", "-q", str(GHP), "gh-pages", check=True)


def fetch():
    _git("fetch", "-q", "origin", BRANCH)


def sync_control():
    """Applies app-side control.json changes to runs/control.json.
    Returns True if a change was applied."""
    r = _git("rev-parse", CONTROL_REF)
    if r.returncode != 0:
        return False  # no control file committed yet
    blob = r.stdout.strip()
    st = _state()
    if blob == st["control_hash"]:
        return False
    content = _git("show", CONTROL_REF, check=True).stdout
    json.loads(content)  # validate before applying
    (RUNS / "control.json").write_text(content)
    st["control_hash"] = blob
    _save_state(st)
    return True


def pending_requests():
    r = _git("ls-tree", "--name-only", REQ_DIR_REF)
    if r.returncode != 0:
        return []
    names = [n for n in r.stdout.split() if n.endswith(".json")]
    done = set(_state()["processed"])
    return [n for n in sorted(names) if n not in done]


def _mark_processed(name):
    st = _state()
    st["processed"] = (st["processed"] + [name])[-500:]
    _save_state(st)


def _push_ghp(message):
    _git("add", "-A", cwd=GHP)
    _git("commit", "-q", "-m", message + COMMIT_TRAILER, cwd=GHP)
    for delay in (0, 2, 4, 8):
        if delay:
            time.sleep(delay)
            _git("pull", "-q", "--rebase", "origin", "gh-pages", cwd=GHP)
        if _git("push", "-q", "origin", "gh-pages", cwd=GHP).returncode == 0:
            return True
    return False


GAME_HARD_CAP = 1200   # seconds; kill a background game that runs longer


def start_request(name, log=print):
    """Launches one requested game as a background subprocess (non-blocking) so
    it never stalls the training loop. Returns a job dict to poll with
    poll_job()."""
    content = _git("show", f"{REQ_DIR_REF}/{name}", check=True).stdout
    req = json.loads(content)
    rid = str(req.get("id") or Path(name).stem)
    req_file = RUNS / f"req_{rid}.json"
    req_file.write_text(content)
    games_dir = GHP / "data" / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    out = games_dir / f"{rid}.json"
    logf = open(RUNS / f"game_{rid}.log", "w")
    proc = subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "game_player.py"),
         "--request", str(req_file), "--out", str(out)],
        stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT))
    log(f"started requested game {rid} in background (pid {proc.pid})")
    return {"name": name, "rid": rid, "proc": proc, "out": out, "logf": logf,
            "req_file": req_file, "req": req, "started": time.time()}


def poll_job(job, log=print):
    """Checks a running game job. Returns True once it is finished (and its
    result/failure has been published + marked processed)."""
    proc, rid = job["proc"], job["rid"]
    if proc.poll() is None:                       # still running
        if time.time() - job["started"] > GAME_HARD_CAP:
            proc.kill(); proc.wait()
            _finish_job(job, error="too slow to finish (timed out)",
                        log=log, verb="killed (exceeded cap)")
            return True
        return False
    if proc.returncode != 0 or not job["out"].exists():
        tail = ""
        try:
            tail = Path(RUNS / f"game_{rid}.log").read_text()[-400:]
        except OSError:
            pass
        _finish_job(job, error="failed to play", log=log,
                    verb=f"FAILED: {tail}")
        return True
    replay = json.loads(job["out"].read_text())
    _finish_job(job, entry={
        "id": rid, "t": time.time(),
        "tribes": replay["tribes"], "seats": replay["seats"],
        "mode": replay["mode"], "result": replay["result"],
        "frames": len(replay["frames"])}, log=log, verb="played and published")
    return True


def _finish_job(job, entry=None, error=None, log=print, verb=""):
    try:
        job["logf"].close()
    except Exception:
        pass
    job["req_file"].unlink(missing_ok=True)
    if entry is None:
        entry = {"id": job["rid"], "error": error,
                 "tribes": job["req"].get("tribes"), "t": time.time()}
    _mark_processed(job["name"])
    _publish_index_entry(entry)
    log(f"game request {job['rid']} {verb}")


def _publish_index_entry(entry):
    idx_path = GHP / "data" / "games" / "index.json"
    try:
        idx = json.loads(idx_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        idx = []
    idx = [e for e in idx if e.get("id") != entry["id"]]
    idx.insert(0, entry)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(idx[:50], separators=(",", ":")))
    _push_ghp(f"Game {entry['id']}")


def publish_data(payload):
    (GHP / "data").mkdir(exist_ok=True)
    (GHP / "data" / "data.json").write_text(
        json.dumps(payload, separators=(",", ":")))
    return _push_ghp(f"Training data (iter {payload['totals']['iterations']})")


def push_checkpoint(ckpt_path, iteration):
    """Commits the current checkpoint to the working branch so the trained
    network survives a container rollback. Best-effort; never raises."""
    try:
        dest = Path(REPO) / "polytopia-rl" / "checkpoints" / "policy.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() != Path(ckpt_path).resolve():
            dest.write_bytes(Path(ckpt_path).read_bytes())
        _git("add", "-f", "polytopia-rl/checkpoints/policy.pt")
        if not _git("diff", "--cached", "--quiet").returncode:
            return  # nothing changed
        _git("commit", "-q", "-m", f"Checkpoint at iteration {iteration}" + COMMIT_TRAILER)
        for delay in (0, 2, 4, 8):
            if delay:
                time.sleep(delay)
                _git("pull", "-q", "--rebase", "origin", BRANCH)
            if _git("push", "-q", "origin", BRANCH).returncode == 0:
                return
    except Exception:
        pass
