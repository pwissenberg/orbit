"""Download the STRING inputs of the ORBIT PPI track from the SPACE Zenodo record.

ORBIT does not recompute anything for STRING (paper Sections 2.1 and 2.8): it rotates the
per-species Node2Vec embeddings that Hu et al. (2025) released with SPACE, anchored on the
eggNOG orthogroups at the eukaryotic root that SPACE also uses. Both come from Zenodo record
15600639 (https://zenodo.org/records/15600639). ``orbit fetch string`` writes exactly the
files ``orbit align`` needs:

    <dest>/euk_seed_groups.json   the 48 SPACE seed species (17 metazoa, 13 fungi, 13 plants, 5 protists)
    <dest>/euks.txt               the 1,322 eukaryotic taxa of the benchmark
    <dest>/seeds.txt              the 48 seeds, one per line, in SPACE's group order
    <dest>/eggnog/2759.tsv.gz     eggNOG orthogroup members at the Eukaryota root
    <dest>/node2vec/<taxid>.h5    unaligned Node2Vec embeddings, 128-d float32

The two archives are large (node2vec.zip 4.8 GB, eggnog.zip 2.8 GB). By default only the
members that are needed are extracted through HTTP range requests, so a small run needs
about 0.7 GB of transfer. Dropped connections are resumed and files that exist are skipped.
Requires only the standard library.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

RECORD = "15600639"
API = f"https://zenodo.org/api/records/{RECORD}"
EGGNOG_LEVEL = "2759"  # Eukaryota, the level SPACE and the paper use for anchors
SEED_GROUPS = ["metazoa", "fungi", "plants", "protists"]
CHUNK = 1 << 22
RETRIES = 8


def log(msg: str) -> None:
    print(msg, flush=True)


def seed_list(groups: dict) -> list[str]:
    """The SPACE seeds as strings, in the group order metazoa, fungi, plants, protists."""
    return [str(t) for g in SEED_GROUPS for t in groups[g]]


def plan_taxa(seeds: list[str], euks: list[str], taxa: list[str], all_: bool) -> list[str]:
    """Taxa whose Node2Vec files are needed: everything with ``all_``, else the seeds plus
    ``taxa`` without duplicates. Raises ``ValueError`` for taxa outside the benchmark."""
    if all_:
        return list(euks)
    known = set(euks)
    unknown = [t for t in taxa if t not in known]
    if unknown:
        raise ValueError(f"not in euks.txt: {unknown}")
    return list(dict.fromkeys(seeds + [str(t) for t in taxa]))


def record_files() -> dict[str, dict]:
    with urllib.request.urlopen(API, timeout=60) as r:
        rec = json.load(r)
    return {f["key"]: f for f in rec["files"]}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_md5: str | None = None, size: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expected_md5 is None or md5_of(dest) == expected_md5:
            log(f"  have {dest}")
            return
        log(f"  {dest} has a wrong checksum, downloading again")
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    for attempt in range(1, RETRIES + 1):
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab" if done else "wb") as out:
                if done and r.status != 206:  # server ignored the resume request: start over
                    out.seek(0)
                    out.truncate()
                    done = 0
                for block in iter(lambda: r.read(CHUNK), b""):
                    out.write(block)
                    done += len(block)
                    if size and done % (CHUNK * 64) < CHUNK:
                        log(f"    {done / 1e9:.2f} / {size / 1e9:.2f} GB")
            break
        except (TimeoutError, OSError) as exc:
            if attempt == RETRIES:
                raise RuntimeError(f"download failed after {RETRIES} attempts: {exc}") from exc
            log(f"    connection dropped at {done / 1e9:.2f} GB ({exc}); resuming, attempt {attempt + 1}")
    if expected_md5 and md5_of(tmp) != expected_md5:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {dest.name}")
    tmp.replace(dest)
    log(f"  wrote {dest}")


class RemoteFile(io.RawIOBase):
    """Read-only, seekable view of a remote file via HTTP range requests.

    ``zipfile`` only needs a handful of reads to parse the central directory and one
    read per extracted member, so this keeps the transfer close to the member sizes.
    """

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, min(self.size, base + offset))
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n == 0:
            return b""
        buf = io.BytesIO()
        got = 0
        for attempt in range(1, RETRIES + 1):
            start = self.pos + got
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{self.pos + n - 1}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise RuntimeError("server did not honour the range request; use --all")
                    for block in iter(lambda: r.read(CHUNK), b""):
                        buf.write(block)
                        got += len(block)
                        if n > 50 * CHUNK and got % (CHUNK * 16) < CHUNK:
                            log(f"    {got / 1e6:.0f} / {n / 1e6:.0f} MB")
                if got == n:
                    break
            except (TimeoutError, OSError) as exc:
                if attempt == RETRIES:
                    raise RuntimeError(f"download failed after {RETRIES} attempts: {exc}") from exc
                log(f"    connection dropped at {got / 1e6:.0f} MB ({exc}); resuming, attempt {attempt + 1}")
        data = buf.getvalue()
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def extract_members(zf: zipfile.ZipFile, wanted: dict[str, Path]) -> None:
    """Extract archive members by basename into the given destination paths."""
    by_name = {Path(i.filename).name: i for i in zf.infolist() if not i.is_dir()}
    missing = [n for n in wanted if n not in by_name]
    if missing:
        raise RuntimeError(f"not in archive: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    for name, dest in wanted.items():
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with zf.open(by_name[name]) as src, open(tmp, "wb") as out:
            shutil.copyfileobj(src, out, CHUNK)
        tmp.replace(dest)  # a file under <dest> is complete or absent, never truncated
        log(f"  wrote {dest}")


def fetch_members(files: dict[str, dict], archive: str, wanted: dict[str, Path], dest_root: Path, full: bool) -> None:
    todo = {n: p for n, p in wanted.items() if not p.exists()}
    if not todo:
        log(f"  have all {len(wanted)} files from {archive}")
        return
    meta = files[archive]
    url, size = meta["links"]["self"], meta["size"]
    if full:
        zpath = dest_root / archive
        log(f"downloading {archive} ({size / 1e9:.1f} GB)")
        download_file(url, zpath, meta["checksum"].removeprefix("md5:"), size)
        with zipfile.ZipFile(zpath) as zf:
            extract_members(zf, todo)
    else:
        log(f"extracting {len(todo)} member(s) of {archive} by range request")
        with zipfile.ZipFile(RemoteFile(url, size)) as zf:  # type: ignore[arg-type]
            extract_members(zf, todo)


def fetch_string(dest: Path, *, taxa: list[str] = (), all_: bool = False, keep_archives: bool = False) -> int:
    """Fetch the STRING inputs into ``dest`` (see the module docstring). Returns 0."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = record_files()
    log(f"Zenodo record {RECORD}: {len(files)} files")

    for key in ("euk_seed_groups.json", "euks.txt"):
        download_file(files[key]["links"]["self"], dest / key, files[key]["checksum"].removeprefix("md5:"))
    groups = json.loads((dest / "euk_seed_groups.json").read_text())
    seeds = seed_list(groups)
    euks = [t.strip() for t in (dest / "euks.txt").read_text().splitlines() if t.strip()]
    (dest / "seeds.txt").write_text("\n".join(seeds) + "\n")
    log(f"  {len(seeds)} seeds -> {dest / 'seeds.txt'}; {len(euks) - len(seeds)} non-seeds in the benchmark")

    wanted = plan_taxa(seeds, euks, [str(t) for t in taxa], all_)
    fetch_members(files, "eggnog.zip",
                  {f"{EGGNOG_LEVEL}.tsv.gz": dest / "eggnog" / f"{EGGNOG_LEVEL}.tsv.gz"}, dest, all_)
    fetch_members(files, "node2vec.zip", {f"{t}.h5": dest / "node2vec" / f"{t}.h5" for t in wanted}, dest, all_)
    if all_ and not keep_archives:
        for a in ("eggnog.zip", "node2vec.zip"):
            (dest / a).unlink(missing_ok=True)

    n = len(list((dest / "node2vec").glob("*.h5")))
    log(f"done: {n} Node2Vec files under {dest / 'node2vec'}; next: orbit align --embeddings {dest / 'node2vec'} "
        f"--orthogroups {dest / 'eggnog' / f'{EGGNOG_LEVEL}.tsv.gz'} --seeds {dest / 'seeds.txt'} "
        f"--reference 9606 --allow-reflection --out results/string")
    return 0
