#!/usr/bin/env zsh
set -euo pipefail

# プロセス名(コマンド名)が ahc[数字3桁] で始まるもの
pattern='^ahc[0-9]{3}'

# pgrep はデフォで「プロセス名」を正規表現マッチ（-f を付けない）
# -l で PIDと名前を出す → awkでPIDだけ抜く
pids=("${(@f)$(pgrep -l $pattern | awk '{print $1}')}")

if (( ${#pids} == 0 )); then
  echo "対象プロセスは見つかりませんでした。"
  exit 0
fi

echo "SIGTERM を送ります: ${pids[@]}"
kill -TERM -- ${pids[@]}

# 少し待ってまだ生きてるやつを強制終了
sleep 1
alive=("${(@f)$(ps -o pid= -p ${^pids} 2>/dev/null | tr -d ' ')}")
if (( ${#alive} )); then
  echo "まだ生きているので SIGKILL: ${alive[@]}"
  kill -KILL -- ${alive[@]}
fi

echo "完了。"
