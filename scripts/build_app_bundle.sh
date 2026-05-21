#!/usr/bin/env bash
# Build the "NFL Forecast.app" double-clickable launcher.
#
# Why not write the .app bundle by hand?
#   On modern macOS (Apple Silicon Sequoia/Sonoma), LaunchServices /
#   amfid refuses to spawn unsigned shell scripts as the main executable
#   of a .app bundle. The launch dies with POSIX 162 ("Launchd job spawn
#   failed") and the icon flashes briefly with no error dialog.
#
#   The reliable, no-Apple-Developer-account workaround is to use an
#   AppleScript "applet" as the main executable. ``osacompile`` produces
#   a properly-signed bundle whose ``applet`` Mach-O is provided by
#   Apple, so AMFI is happy. Our actual launcher logic lives in
#   ``Contents/Resources/launcher.sh`` and is invoked from the .app via
#   ``do shell script``.
#
# Re-run this script any time you edit launcher.sh or want a fresh bundle.

set -euo pipefail

ROOT="$( cd "$( dirname "$0" )/.." && pwd )"
APP="${ROOT}/NFL Forecast.app"
SCRIPT="${ROOT}/scripts/nfl_forecast_applescript.applescript"
LAUNCHER="${ROOT}/scripts/nfl_forecast_launcher.sh"
ICON_SRC="${ROOT}/NFL Forecast.icns"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Missing ${SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${LAUNCHER}" ]]; then
  echo "Missing ${LAUNCHER}" >&2
  exit 1
fi

# Wipe any previous bundle.
rm -rf "${APP}"

# Compile the AppleScript into a .app bundle.
osacompile -o "${APP}" "${SCRIPT}"

# Drop our bash launcher into Resources/ where the AppleScript reads it.
cp "${LAUNCHER}" "${APP}/Contents/Resources/launcher.sh"
chmod +x "${APP}/Contents/Resources/launcher.sh"

# Patch Info.plist: set bundle id, display name, and turn on Dark Mode support.
INFO_PLIST="${APP}/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier com.local.nflforecast" "${INFO_PLIST}" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string com.local.nflforecast" "${INFO_PLIST}"
/usr/libexec/PlistBuddy -c "Set :CFBundleName NFL Forecast" "${INFO_PLIST}" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Add :CFBundleName string 'NFL Forecast'" "${INFO_PLIST}"
/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName NFL Forecast" "${INFO_PLIST}" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Add :CFBundleDisplayName string 'NFL Forecast'" "${INFO_PLIST}"
/usr/libexec/PlistBuddy -c "Set :NSHighResolutionCapable true" "${INFO_PLIST}" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Add :NSHighResolutionCapable bool true" "${INFO_PLIST}"
/usr/libexec/PlistBuddy -c "Set :NSRequiresAquaSystemAppearance false" "${INFO_PLIST}" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Add :NSRequiresAquaSystemAppearance bool false" "${INFO_PLIST}"
/usr/libexec/PlistBuddy -c "Set :LSUIElement false" "${INFO_PLIST}" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Add :LSUIElement bool false" "${INFO_PLIST}"

# Replace the generic AppleScript droplet icon with our brand icon if available.
if [[ -f "${ICON_SRC}" ]]; then
  cp "${ICON_SRC}" "${APP}/Contents/Resources/applet.icns"
fi

# Ad-hoc sign so LaunchServices is happy on Apple Silicon.
codesign --force --deep --sign - "${APP}"

# Re-register with LaunchServices so the cached "broken bundle" entry is cleared.
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister
"${LSREGISTER}" -f "${APP}" >/dev/null 2>&1 || true

echo "Built ${APP}"
