#!/usr/bin/env bash
# Installs a user LaunchAgent so Pictor starts at login and restarts if it dies.
# Usage: scripts/install-launchagent.sh [--uninstall]
set -euo pipefail
LABEL=com.yv17labs.pictor
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
if [[ "${1:-}" == "--uninstall" ]]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed $PLIST"
  exit 0
fi
UV="$(command -v uv)"
mkdir -p "$HOME/Library/LaunchAgents"
sed -e "s#__UV__#$UV#g" -e "s#__PROJECT__#$PROJECT#g" -e "s#__HOME__#$HOME#g" \
  "$PROJECT/scripts/$LABEL.plist" > "$PLIST"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $PLIST — logs: ~/Library/Logs/pictor.log"
echo "health: curl -s localhost:8091/health"
