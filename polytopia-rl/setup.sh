#!/usr/bin/env bash
# Rebuilds the working tree at /root/polytopia-rl after a fresh container.
# Recovers from a rollback in one command: restores code from the branch,
# rebuilds the venv + Java classes, and restarts the training daemon from the
# last checkpoint committed to git.
set -e
ROOT=/root/polytopia-rl
REPO=/home/user/AutoGPT
BRANCH=claude/polytopia-rl-setup-ayk6ia

echo "==> syncing repo to latest branch"
git -C "$REPO" fetch -q origin "$BRANCH"
git -C "$REPO" reset --hard "origin/$BRANCH"

echo "==> restoring code into $ROOT"
mkdir -p "$ROOT"
cp -r "$REPO"/polytopia-rl/. "$ROOT"/ 2>/dev/null || true

echo "==> cloning Tribes if missing"
[ -d "$ROOT/Tribes/src" ] || git clone -q https://github.com/GAIGResearch/Tribes.git "$ROOT/Tribes"

echo "==> patching Tribes for CONQUEST mode (MAX_TURNS settable at runtime)"
# The upstream MAX_TURNS is a compile-time final constant; CONQUEST needs to
# raise it, so make it a mutable public static field. Idempotent.
sed -i 's/^    static final int MAX_TURNS = 30;/    public static int MAX_TURNS = 30;/' \
  "$ROOT/Tribes/src/core/Constants.java"

echo "==> compiling Tribes + shim"
cd "$ROOT/Tribes"
# Always rebuild so the CONQUEST patch (and any re-clone) is reflected.
rm -rf out; mkdir -p out; javac -cp lib/json.jar -d out $(find src -name '*.java') 2>/dev/null
mkdir -p "$ROOT/java/out"
javac -cp out:lib/json.jar -d "$ROOT/java/out" "$ROOT/java/src/core/game/RLGameRunner.java"

echo "==> python venv"
cd "$ROOT"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q numpy jpype1 >/dev/null 2>&1 || true
.venv/bin/python -c "import torch" 2>/dev/null || \
  .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1

echo "==> gh-pages worktree"
git -C "$REPO" worktree prune
rm -rf "$ROOT/ghp"
git -C "$REPO" fetch -q origin gh-pages
git -C "$REPO" worktree add -q "$ROOT/ghp" gh-pages

echo "==> starting daemon"
mkdir -p "$ROOT/runs"
pkill -f train_daemon.py 2>/dev/null || true
sleep 1
cd "$ROOT"
nohup .venv/bin/python train_daemon.py > runs/daemon.log 2>&1 &
sleep 3
echo "==> done. daemon pid: $(pgrep -f train_daemon.py || echo NONE)"
