#!/usr/bin/env bash
# Build and test the current branch on a remote machine, and bring the
# result back as a table: what passed, what failed, and what skipped with
# the reason it gave. A skip is not a pass, and the table says which is
# which.
#
#   scripts/remote-test.sh [user@host] [preset] [branch]
#
# Defaults: ubuntu@10.47.214.94, linux-x86_64-debug, the current branch.
# LRT_BACKEND (cuda, vulkan) is passed through when set, so one suite can
# be run once a backend at a time. The remote's ctest log lands in
# build/remote/<preset>/LastTest.log.
set -euo pipefail

host="${1:-ubuntu@10.47.214.94}"
preset="${2:-linux-x86_64-debug}"
branch="${3:-$(git rev-parse --abbrev-ref HEAD)}"
remote_dir="${LRT_REMOTE_DIR:-~/inn/lucabRTrender}"
backend="${LRT_BACKEND:-}"

echo "== ${host}: ${branch} with preset ${preset}${backend:+ on ${backend}}"
ssh "${host}" "cd ${remote_dir} && git fetch -q origin && git checkout -q ${branch} && git reset -q --hard origin/${branch} && git submodule update -q --init --recursive && git log --oneline -1 && (cmake --preset ${preset} >/dev/null) && cmake --build --preset ${preset} 2>&1 | grep -E 'error:|FAILED' | head -20; echo '--- built ---'; ${backend:+LRT_BACKEND=${backend}} ctest --preset ${preset} 2>&1 | grep -E '^ *[0-9]+/[0-9]+ Test|tests passed|Total Test' " | tee /tmp/lrt-remote-ctest.txt

mkdir -p "build/remote/${preset}"
scp -q "${host}:${remote_dir}/build/${preset}/Testing/Temporary/LastTest.log" "build/remote/${preset}/LastTest.log" || true

# The table: one line a test, with the skip's reason from the log.
python3 - "${preset}" <<'PY2'
import re, sys
preset = sys.argv[1]
lines = open('/tmp/lrt-remote-ctest.txt', encoding='utf-8', errors='replace').read().splitlines()
try:
    log = open(f'build/remote/{preset}/LastTest.log', encoding='utf-8', errors='replace').read()
except OSError:
    log = ''
# LastTest.log opens each test with "N/M Testing: NAME"; Catch2's skip reads
# "SKIPPED:", "explicitly with message:", then the message indented, wrapped
# over as many lines as it takes, up to a blank line.
reasons = {}
blocks = re.split(r'^\d+/\d+ Testing: ', log, flags=re.M)
for block in blocks[1:]:
    name, _, body = block.partition('\n')
    skip = re.search(r'SKIPPED:\s*\n(?:explicitly with message:\s*\n)?((?:[ \t]+\S.*\n?)+)', body)
    if skip:
        reasons[name.strip()] = ' '.join(part.strip() for part in skip.group(1).splitlines())
passed = failed = skipped = 0
for line in lines:
    m = re.match(r'\s*\d+/\d+ Test\s+#\d+: (.+?) \.+\s*(\**)(Passed|Failed|Skipped|Timeout|Not Run)', line)
    if not m:
        continue
    name, _, state = m.groups()
    if state == 'Passed':
        passed += 1
    elif state == 'Skipped':
        skipped += 1
        print(f'  skipped  {name}  -- {reasons.get(name, "(no reason recorded)")}')
    else:
        failed += 1
        print(f'  FAILED   {name}')
print(f'== {passed} passed, {failed} failed, {skipped} skipped')
PY2
