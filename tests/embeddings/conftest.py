"""Keep the embedding tests off the network unless a download is asked for."""

import os

if not os.environ.get("COPERNICUS_TEST_DOWNLOAD"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
