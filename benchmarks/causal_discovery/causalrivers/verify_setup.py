"""Download the pinned archive and independently compare frozen product inputs."""

import hashlib
import json
import shutil
import urllib.request
import zipfile

from tools.integrity import BENCHMARK_ROOT
from tools.provenance import file_record, sha256

URL = "https://github.com/CausalRivers/benchmark/releases/download/First_release/product.zip"
DIGEST = "4f779fa9222653af092046eacba39a955cc65638bfb70fb7e31a32d9d8373b55"


def main():
    archive = BENCHMARK_ROOT / "product.zip"
    if not archive.exists():
        print(f"Downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, archive)
    actual = sha256(archive)
    if actual != DIGEST:
        raise ValueError(f"Archive digest mismatch: {actual}")
    records = []
    with zipfile.ZipFile(archive) as zipped:
        for filename in (
            "rivers_east_germany.p",
            "rivers_bavaria.p",
            "rivers_flood.p",
            "rivers_ts_east_germany.csv",
        ):
            candidates = [name for name in zipped.namelist() if name.rsplit("/", 1)[-1] == filename]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous or missing archive member: {filename}")
            target = BENCHMARK_ROOT / "product" / filename
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(candidates[0]) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
            digest = hashlib.sha256()
            with zipped.open(candidates[0]) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            record = file_record("product/" + filename)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Frozen product differs from pinned archive: {filename}")
            records.append({**record, "archive_member": candidates[0]})
            print(f"Verified archive member {filename}: {record['sha256']}", flush=True)
    output = BENCHMARK_ROOT / "evidence/review-232/setup.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"archive_url": URL, "archive_sha256": actual, "verified_products": records}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print("Pinned archive and all four selected frozen products match.", flush=True)


if __name__ == "__main__":
    main()
