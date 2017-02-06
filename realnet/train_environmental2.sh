#!/usr/bin/env sh
set -e
#!sourcecaffe

/u/vis/x1/redmunds/git/caffeGPU/caffe/build/tools/caffe train --solver=/u/vis/x1/redmunds/git/sound/environmental/realnodropout/sound_solver2.prototxt $@
