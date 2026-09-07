#!/usr/bin/env bash
set -euo pipefail

RUNTIME_CONFIG="${STACKCHAN_RUNTIME_CONFIG:-/home/ernest/stackchan-run/manifest/config/config.yaml}"
SETTINGS_TOKEN="${STACKCHAN_SETTINGS_TOKEN:-}"
DEVICE_IP="${STACKCHAN_DEVICE_IP:-192.168.219.136}"
SINCE="${STACKCHAN_VERIFY_SINCE:-15 minutes ago}"

if [[ -z "${SETTINGS_TOKEN}" && -r "${RUNTIME_CONFIG}" ]]; then
  SETTINGS_TOKEN="$(sed -nE 's/^[[:space:]]*settings_auth_token:[[:space:]]*"?([^"[:space:]#]+)"?.*$/\1/p' "${RUNTIME_CONFIG}" | head -n 1)"
fi
if [[ -z "${SETTINGS_TOKEN}" ]]; then
  echo "settings token is unavailable; set STACKCHAN_SETTINGS_TOKEN or STACKCHAN_RUNTIME_CONFIG" >&2
  exit 1
fi

echo "== service =="
systemctl is-active stackchan.service
systemctl show stackchan.service -p ActiveState -p SubState -p MainPID --no-pager

echo "== ports =="
ss -ltnp | grep -E ':12800|:8099'

echo "== runtime settings =="
curl -fsS --max-time 5 -H "Authorization: Bearer ${SETTINGS_TOKEN}" \
  http://127.0.0.1:8099/api/settings > /tmp/stackchan-eve-settings.json
python3 - <<'PY'
import json
from pathlib import Path

values = json.loads(Path("/tmp/stackchan-eve-settings.json").read_text(encoding="utf-8"))
required = {
    "autonomous_actions_enabled": "false",
    "face_contact_enabled": "false",
    "conversation_idle_seconds": "15",
    "stt_language": "ko",
    "stackchan_motion_speed": "180",
    "stackchan_motion_step_delay_ms": "220",
}
for key, want in required.items():
    got = values.get(key)
    print(f"{key}={got}")
    if got != want:
        raise SystemExit(f"{key}: got {got!r}, want {want!r}")
if "happy" not in values.get("stackchan_expression_colors", ""):
    raise SystemExit("stackchan_expression_colors missing happy color")
print("settings_ok=true")
PY

echo "== no unintended autonomy/camera logs =="
if journalctl -u stackchan.service --since "${SINCE}" --no-pager |
  grep -E '\[AUTONOMY\]|head heartbeat|automatic camera|\[FACE\]' >/tmp/stackchan-eve-unwanted.log; then
  cat /tmp/stackchan-eve-unwanted.log
  exit 1
fi
echo "unwanted_logs=false"

echo "== device network =="
if ping -c 1 -W 1 "${DEVICE_IP}" >/dev/null; then
  echo "device_ping=ok"
else
  echo "device_ping=missing"
fi

echo "== recent device MCP/tools =="
journalctl -u stackchan.service --since "${SINCE}" --no-pager |
  grep -E '\[WS\] device|ready tools|\[MCP\]' || true
