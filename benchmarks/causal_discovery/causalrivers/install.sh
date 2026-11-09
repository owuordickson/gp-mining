#!/usr/bin/env bash
set -euo pipefail

# 1. Initialize Conda runtime configuration mapping
conda env create -f causal_rivers_core.yml

# 2. Pinned asset definitions
PINNED_URL="https://github.com/CausalRivers/benchmark/releases/download/First_release/product.zip"
EXPECTED_SHA="4F779FA9222653AF092046EACBA39A955CC65638BFB70FB7E31A32D9D8373B55"
TARGET_FILE="product.zip"

echo "Downloading pinned benchmark archive..."
wget -O "$TARGET_FILE" "$PINNED_URL"

# 3. Integrity Verification Layer (Enforces #211 safeguard constraints)
echo "Verifying SHA-256 integrity signature..."
echo "$EXPECTED_SHA  $TARGET_FILE" | sha256sum --check --status

if [ $? -ne 0 ]; then
    echo "ERROR: Cryptographic signature mismatch! The downloaded file is untrusted or corrupted." >&2
    exit 1
fi

echo "Integrity check passed. Unzipping asset safely..."
unzip product
rm product.zip

echo "conda environment is installed as causalrivers. Done."
