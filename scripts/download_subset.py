"""Download sequences 0000 and 0011 of the KITTI tracking benchmark (training split).

The images add up to ~15 GB, so they are pulled out of the remote zip with range
requests (remotezip) instead of downloading the whole archive. Labels and calib
are a few MB and are downloaded in full.
"""

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from remotezip import RemoteZip

BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
IMAGE_ZIP = f"{BASE_URL}/data_tracking_image_2.zip"
LABEL_ZIP = f"{BASE_URL}/data_tracking_label_2.zip"
CALIB_ZIP = f"{BASE_URL}/data_tracking_calib.zip"

SEQUENCES = ["0000", "0011"]
DEST = Path(__file__).resolve().parents[1] / "data" / "kitti_tracking"


def download_images(dest):
    with RemoteZip(IMAGE_ZIP) as z:
        names = z.namelist()
        for seq in SEQUENCES:
            prefix = f"training/image_02/{seq}/"
            members = sorted(n for n in names if n.startswith(prefix) and n.endswith(".png"))
            print(f"images for sequence {seq}: {len(members)} frames")
            for i, name in enumerate(members, 1):
                out = dest / name
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"\r  {i}/{len(members)}", end="", flush=True)
            print()


def download_whole_zip(url, members, dest):
    print(f"downloading {url.rsplit('/', 1)[1]}")
    with urllib.request.urlopen(url) as response:
        blob = response.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in members:
            z.extract(name, dest)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    download_whole_zip(LABEL_ZIP, [f"training/label_02/{s}.txt" for s in SEQUENCES], DEST)
    download_whole_zip(CALIB_ZIP, [f"training/calib/{s}.txt" for s in SEQUENCES], DEST)
    download_images(DEST)
    print(f"done: {DEST}")


if __name__ == "__main__":
    main()
