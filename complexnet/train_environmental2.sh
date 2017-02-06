#!/usr/bin/env sh
set -e

/u/vis/x1/redmunds/git/GPUCOMPLEXNEW/caffe/build/tools/caffe train --solver=/u/vis/x1/redmunds/git/sound/environmental/complexnet/sound_solver2.prototxt $@

