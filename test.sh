#!/bin/sh
# Sun Dec  7 12:38:32 AM CST 2025
# The test file referenced here is available from:
# https://upload.wikimedia.org/wikipedia/commons/4/47/Acompanyament_Tema_de_Lucil%C2%B7la.wav
# It's ~15.69 MB, so it was not included in the repository.

# The operational default is to test the left channel.
export CHANNEL="${1:-left}"

export CPS=32
export SAMPLES_PER_CHIP=8
export CARRIER_HZ=11025
export IN_FILE=Acompanyament_Tema_de_Lucil·la.wav
#export IN_FILE=silence.wav

echo "Test message from test.sh" > payload.bin

# silence.wav is generated thus:
ffmpeg -loglevel quiet -y -f lavfi -i \
       anullsrc=channel_layout=stereo:sample_rate=44100 \
       -t 30 -q:a 9 -acodec pcm_s16le silence.wav

cargo run -- \
  --mode embed \
  --input $IN_FILE \
  --output stereo_with_dsss.wav \
  --payload payload.bin \
  --dsss-dbfs -75 \
  --delay-fraction 0.25 \
  --seed "test-seed" \
  --spreading-factor $CPS \
  --samples-per-chip $SAMPLES_PER_CHIP \
  --carrier-freq $CARRIER_HZ \
  --visualize

#if [ $? -ne 0 ]; then
	#exit 1
#fi

echo

cargo run -- \
  --mode decode-wav \
  --input stereo_with_dsss.wav \
  --output decoded.bin \
  --seed "test-seed" \
  --spreading-factor $CPS \
  --samples-per-chip $SAMPLES_PER_CHIP \
  --carrier-freq $CARRIER_HZ \
  --channel $CHANNEL

