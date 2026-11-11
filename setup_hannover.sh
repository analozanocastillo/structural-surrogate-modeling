#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ML_STRUCTURAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_URL="${ML_STRUCTURAL_DATA_URL:-https://data.uni-hannover.de/dataset/39b91d0b-5b0d-4fb8-bdd9-28c4d942165c/resource/6350fc21-2dfd-4067-ac02-db118348c996/download/cantileverbeam.zip}"
PDF_URL="${ML_STRUCTURAL_PDF_URL:-https://data.uni-hannover.de/dataset/39b91d0b-5b0d-4fb8-bdd9-28c4d942165c/resource/7d957d42-8384-4737-85dc-8b35bc1e025b/download/readme.pdf}"

mkdir -p "$PROJECT_DIR/data/raw" "$PROJECT_DIR/data/extracted"

fetch_if_missing() {
  local url="$1" path="$2"
  if [[ -s "$path" ]]; then
    echo "Using existing $(basename "$path")"
    return
  fi
  echo "Downloading $(basename "$path") from Hannover..."
  local partial="${path}.part"
  if ! curl --fail --silent --show-error --location --retry 3 --connect-timeout 30 --output "$partial" "$url"; then
    rm -f "$partial"
    echo "Download failed: $url" >&2
    exit 1
  fi
  mv "$partial" "$path"
}

fetch_if_missing "$DATA_URL" "$PROJECT_DIR/data/raw/CantileverBeam.zip"
fetch_if_missing "$PDF_URL" "$PROJECT_DIR/data/raw/ReadMe.pdf"

export PROJECT_DIR
python3 - <<'PY'
from hashlib import sha256
from pathlib import Path
from shutil import copyfileobj
from zipfile import ZipFile
import os

root = Path(os.environ['PROJECT_DIR'])
raw = root / 'data/raw'
destination = (root / 'data/extracted').resolve()
zip_path = raw / 'CantileverBeam.zip'

with ZipFile(zip_path) as archive:
    broken = archive.testzip()
    if broken is not None:
        raise RuntimeError(f'Corrupt ZIP member: {broken}')
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise RuntimeError(f'Unsafe ZIP path: {member.filename}')
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif target.exists():
            if target.stat().st_size != member.file_size:
                raise RuntimeError(f'Existing extracted file has the wrong size: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open('xb') as output:
                copyfileobj(source, output)

required = [
    destination / 'Code_Veröffentlichung/Data/BeamData/LiveBeamData.mat',
    destination / 'Code_Veröffentlichung/Data/ModalProperties/LiveBeamModalData.mat',
    destination / 'Code_Veröffentlichung/Subclass/MassChanges.m',
]
if not all(path.is_file() for path in required):
    raise RuntimeError('The official archive is missing a required Hannover file.')

def file_sha256(path):
    digest = sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

with (raw / 'SHA256SUMS.txt').open('w') as output:
    for name in ('CantileverBeam.zip', 'ReadMe.pdf'):
        path = raw / name
        output.write(f'{file_sha256(path)}  {name}\n')
print('Hannover dataset ready:', destination)
PY
