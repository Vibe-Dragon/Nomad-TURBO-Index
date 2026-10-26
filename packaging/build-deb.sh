#!/bin/sh
# Build the Debian package. This does not install it, does not start
# indexing, and does not delete vectors, ZIMs, or uploads.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
VERSION=$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$ROOT/nomad_turbo/__init__.py")
if [ -z "$VERSION" ]; then
  echo "could not read the package version" >&2
  exit 1
fi
OUT=${1:-"$ROOT/dist/nomad-turbo_${VERSION}_all.deb"}
STAGE=$(mktemp -d)
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

LIB="$STAGE/usr/lib/nomad-turbo"
mkdir -p "$LIB" "$STAGE/usr/bin" \
  "$STAGE/usr/share/applications" \
  "$STAGE/usr/share/icons/hicolor/256x256/apps" \
  "$STAGE/usr/share/pixmaps" \
  "$STAGE/DEBIAN"

cp -a "$ROOT/nomad_turbo" "$LIB/nomad_turbo"
rm -rf "$LIB/nomad_turbo/__pycache__"
find "$LIB" -name '*.pyc' -delete
cp -a "$ROOT/worker" "$LIB/worker"
cat > "$LIB/nomad-turbo" << 'EOF'
#!/bin/sh
# Installed entry. The desktop file calls /usr/bin/nomad-turbo, which is this file.
set -eu
export PYTHONPATH="/usr/lib/nomad-turbo${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m nomad_turbo "$@"
EOF
cp "$LIB/nomad-turbo" "$STAGE/usr/bin/nomad-turbo"
chmod 0755 "$LIB/nomad-turbo" "$STAGE/usr/bin/nomad-turbo"

cp "$ROOT/nomad_turbo/assets/turbo-index.png" \
  "$STAGE/usr/share/icons/hicolor/256x256/apps/nomad-turbo.png"
cp "$ROOT/nomad_turbo/assets/turbo-index.png" \
  "$STAGE/usr/share/pixmaps/nomad-turbo.png"
cp "$ROOT/packaging/nomad-turbo.desktop" \
  "$STAGE/usr/share/applications/nomad-turbo.desktop"
chmod 0644 "$STAGE/usr/share/applications/nomad-turbo.desktop"

cat > "$STAGE/DEBIAN/control" << EOF
Package: nomad-turbo
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: all
Maintainer: NOMAD Turbo <nomad@local>
Depends: python3 (>= 3.10), python3-gi, gir1.2-gtk-3.0
Description: Hardware-scaled indexer for a stock Project NOMAD install
 Nomad TURBO Index runs outside the NOMAD admin container.
 It detects the machine it is on and sizes one embed call from that hardware.
 A specific graphics card is not required.
EOF

mkdir -p "$(dirname "$OUT")"
# mktemp is mode 0700. The archive's "." member is that directory. Leave it
# 0755 so extracting the package cannot tighten the destination.
chmod 0755 "$STAGE"
find "$STAGE" -type d -exec chmod 0755 {} +
chmod 0644 "$STAGE/DEBIAN/control"
dpkg-deb --root-owner-group -Zxz --build "$STAGE" "$OUT"
echo "$OUT"
