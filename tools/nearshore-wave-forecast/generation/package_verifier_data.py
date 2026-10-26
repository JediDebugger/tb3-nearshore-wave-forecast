"""I package the fixed inputs for the separate verifier image."""

import gzip
from pathlib import Path
import tarfile


def main():
    task = Path(__file__).resolve().parents[3] / "tasks/nearshore-wave-forecast"
    source = task / "environment/data"
    output = task / "tests/fixed-data.tar.gz"
    with output.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                info = tarfile.TarInfo("data/" + path.relative_to(source).as_posix())
                info.size = path.stat().st_size
                info.mode = 0o644
                with path.open("rb") as content:
                    archive.addfile(info, content)
    print(f"Packaged {output.stat().st_size} bytes into {output}")


if __name__ == "__main__":
    main()
