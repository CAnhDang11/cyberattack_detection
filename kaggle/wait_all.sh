#!/bin/zsh
# Wait for every kernel pushed with --no-wait, download each output, then clear the list.
cd "$(dirname $0)/.."
KAGGLE=.venv/bin/kaggle; P=kaggle/_build/pending.txt
[ -s $P ] || { echo "nothing pending"; exit 0; }
typeset -A done
while true; do
  left=0
  while read -r KID OUT; do
    [ -n "${done[$KID]}" ] && continue
    st=$($KAGGLE kernels status $KID 2>&1 | tail -1)
    case "$st" in
      *COMPLETE*|*ERROR*|*CANCEL*)
        mkdir -p $OUT; $KAGGLE kernels output $KID -p $OUT -q >/dev/null 2>&1 || true
        done[$KID]=1; echo "$(date +%T) $KID -> ${st##*.} ; outputs in $OUT";;
      *) left=$((left+1)); echo "$(date +%T) $KID ${st##*.}";;
    esac
  done < $P
  [ $left = 0 ] && break; sleep 60
done
mv $P $P.$(date +%m%d%H%M).done; echo "all collected"
