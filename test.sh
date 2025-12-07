#!/bin/sh
# Sun Dec  7 12:38:32 AM CST 2025
# The test file referenced here is available from:
# https://upload.wikimedia.org/wikipedia/commons/4/47/Acompanyament_Tema_de_Lucil%C2%B7la.wav
# It's ~15.69 MB, so it was not included in the repository.

echo "Test message from test.sh" > payload.bin

cargo run -- \
  --mode embed \
  --input Acompanya* \
  --output stereo_with_dsss.wav \
  --payload payload.bin \
  --dsss-dbfs -20 \
  --delay-fraction 0.5 \
  --seed "test-seed" \
  --spreading-factor 128 \
  --samples-per-chip 4 \
  --carrier-freq 16000 \
  --visualize

echo

cargo run -- \
  --mode decode-wav \
  --input stereo_with_dsss.wav \
  --output decoded.bin \
  --seed "test-seed" \
  --spreading-factor 128 \
  --samples-per-chip 4 \
  --carrier-freq 16000 \
  --channel left

