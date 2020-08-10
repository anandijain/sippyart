#!/bin/bash
# TEST_PATH="data/072520.wav"

wavtovid() {
    julia src/vid.jl $1
    ffmpeg -r 60 -f image2 -i data/imgs/%d.png -i $1 -vcodec libx264 -crf 100 out.mp4
}

# wavtovid $TEST_PATH

wavtovid $1