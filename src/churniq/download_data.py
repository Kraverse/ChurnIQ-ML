from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
OUTPUT = Path("data/raw/Telco-Customer-Churn.csv")


def download() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=30) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Dataset checksum mismatch: {digest}")
    OUTPUT.write_bytes(data)
    return OUTPUT


if __name__ == "__main__":
    print(f"Downloaded {download()}")
