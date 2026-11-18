#!/bin/zsh
# Push one experiment to Kaggle as its own PRIVATE GPU kernel.
#   zsh kaggle/run_exp.sh --exp b1|fixed|e3_10class|c2b_8class|b1_10class|fixed_10class|mlp_10class [--seed N] [--config FILE] [--accelerator NvidiaTeslaT4] [--smoke] [--no-wait] [--retry]
#   FILE: b1_best.json ({"config": {...}}) or a plain JSON dict of run_xgb args.
#   --no-wait: push and register in kaggle/_build/pending.txt; collect later with kaggle/wait_all.sh
set -e
cd "$(dirname $0)/.."
KAGGLE=.venv/bin/kaggle; PY=.venv/bin/python
EXP=b1; SEED=42; CONFIG=''; SMOKE=0; WAIT=1; ACC=NvidiaTeslaT4; RETRY=0
while [ $# -gt 0 ]; do case $1 in
  --exp) EXP=$2; shift;; --seed) SEED=$2; shift;; --config) CONFIG=$2; shift;;
  --accelerator) ACC=$2; shift;; --smoke) SMOKE=1;; --retry) RETRY=1;; --no-wait) WAIT=0;; *) echo "unknown arg $1"; exit 1;; esac; shift; done
case $EXP in b1|b1_10class|mlp_10class) ;; *) [ -n "$CONFIG" ] || { echo "--exp $EXP needs --config"; exit 1; };; esac
B=kaggle/_build; mkdir -p $B/code
# Metadata files use the placeholder {user}: every member pushes to their OWN Kaggle account
# (KAGGLE_USER env, else the username of the configured Kaggle CLI).
KUSER=${KAGGLE_USER:-$($KAGGLE config view 2>/dev/null | sed -n 's/^- username: //p')}
[ -n "$KUSER" ] || { echo "Kaggle username unknown: set KAGGLE_USER or configure the Kaggle CLI"; exit 1; }
sed "s/{user}/$KUSER/g" kaggle/dataset-metadata.json > $B/code/dataset-metadata.json
sed "s/{user}/$KUSER/g" kaggle/kernel-metadata.json > $B/kernel-metadata.json
BASE=$($PY -c "import json;print(json.load(open('$B/kernel-metadata.json'))['id'])")
DID=$($PY -c "import json;print(json.load(open('$B/code/dataset-metadata.json'))['id'])")
SLUG=${BASE#*/}; OWNER=${BASE%%/*}
NAME=$SLUG; [ "$EXP" = b1 ] && [ "$SEED" = 42 ] || NAME=$SLUG-${EXP//_/-}-s$SEED
[ $SMOKE = 1 ] && NAME=$NAME-smoke
KID=$OWNER/$NAME; K=$B/kernel_$NAME; rm -rf $K; mkdir -p $K

# 1) code dataset: new version only when unsw/ changed
zip -qr $B/code/kaggle_code.zip unsw NUSW-NB15_features.csv -x '*/__pycache__/*'
HASH=$(cat unsw/*.py NUSW-NB15_features.csv | shasum | cut -c1-16)
if [ "$(cat $B/code_hash 2>/dev/null)" != "$HASH" ]; then
  if $KAGGLE datasets status $DID >/dev/null 2>&1; then $KAGGLE datasets version -p $B/code -m "code $HASH" -q
  else $KAGGLE datasets create -p $B/code -q; fi
  sleep 5; echo "waiting for dataset $DID..."
  until [ "$($KAGGLE datasets status $DID 2>/dev/null)" = "ready" ]; do sleep 10; done
  echo $HASH > $B/code_hash
else echo "code unchanged ($HASH), reuse dataset version"; fi

# 2) notebook copy with the experiment baked in
$PY - "$EXP" "$SEED" "$SMOKE" "$CONFIG" "$K" "$NAME" <<'PY'
import json, sys
exp, seed, smoke, cfgfile, kdir, name = sys.argv[1:]
cfg = 'null'
if cfgfile:
    c = json.load(open(cfgfile)); c = c.get('config', c)
    cfg = json.dumps(c)
nb = json.load(open('kaggle/kaggle_run.ipynb'))
subs = {"SPLIT_SEED = 42 ": f"SPLIT_SEED = {seed} ",
        "SMOKE     = os.environ.get('SMOKE') == '1'": f"SMOKE     = {smoke == '1'} or os.environ.get('SMOKE') == '1'",
        "EXPERIMENT = os.environ.get('EXPERIMENT', 'b1')": f"EXPERIMENT = {exp!r}",
        "CONFIG     = json.loads(os.environ.get('CONFIG_JSON', 'null'))": f"CONFIG     = json.loads({cfg!r})"}
hit = {k: 0 for k in subs}
for c in nb['cells']:
    s = ''.join(c['source'])
    for a, b in subs.items():
        if a in s: s = s.replace(a, b); hit[a] += 1
    c['source'] = s.splitlines(True)
assert all(hit.values()), hit
json.dump(nb, open(f'{kdir}/kaggle_run.ipynb', 'w'), indent=1, ensure_ascii=False)
m = json.load(open('kaggle/_build/kernel-metadata.json'))
m['id'] = m['id'].split('/')[0] + '/' + name; m['title'] = name
json.dump(m, open(f'{kdir}/kernel-metadata.json', 'w'), indent=2)
PY
# Kaggle allows 2 concurrent batch GPU sessions; with --retry, wait for a free slot instead of failing
while true; do
  PUSH=$($KAGGLE kernels push -p $K --accelerator $ACC 2>&1); echo "$PUSH"
  case "$PUSH" in
    *"successfully pushed"*) break;;
    *"session count"*) [ $RETRY = 1 ] || { echo "GPU slots full; rerun later or use --retry"; exit 2; }
                        echo "$(date +%T) GPU slots full, retry in 2 min"; sleep 120;;
    *) echo "push failed"; exit 1;;
  esac
done
OUT=experiments/kaggle/$(date +%m%d_%H%M)_${NAME}
if [ $WAIT = 0 ]; then echo "$KID $OUT" >> $B/pending.txt; echo "pushed $KID (no wait) -> $OUT"; exit 0; fi

# 3) wait + download
while true; do
  st=$($KAGGLE kernels status $KID 2>&1 | tail -1); echo "$(date +%T) $st"
  case "$st" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo FAILED; break;; esac; sleep 60
done
mkdir -p $OUT && $KAGGLE kernels output $KID -p $OUT -q || true
echo "outputs in $OUT"
