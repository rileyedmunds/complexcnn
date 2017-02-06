
from visualize_caffe import *
import sys
 
# Make sure caffe can be found
sys.path.append('/u/vis/x1/redmunds/git/GPUCOMPLEXNEW/caffe/build/tools/caffe/python')
 
import caffe
# Load model
net = caffe.Net('/home/smistad/vessel_net/deploy.prototxt',
                '/home/smistad/vessel_net/snapshot_iter_3800.caffemodel',
                caffe.TEST)

net100 = caffe.Net('/u/vis/x1/redmunds/git/sound/environmental/realnodropout/sound_train2.prototxt',
                '/u/vis/x1/redmunds/git/sound/environmental/realnodropout_iter_100.caffemodel',
                caffe.TEST)

net300 = caffe.Net('/u/vis/x1/redmunds/git/sound/environmental/realnodropout/sound_train2.prototxt',
                '/u/vis/x1/redmunds/git/sound/environmental/realnodropout_iter_300.caffemodel',
                caffe.TEST)

net400 = caffe.Net('/u/vis/x1/redmunds/git/sound/environmental/realnodropout/sound_train2.prototxt',
                '/u/vis/x1/redmunds/git/sound/environmental/realnodropout_iter_400.caffemodel',
                caffe.TEST)

net500 = caffe.Net('/u/vis/x1/redmunds/git/sound/environmental/realnodropout/sound_train2.prototxt',
                '/u/vis/x1/redmunds/git/sound/environmental/realnodropout_iter_500.caffemodel',
                caffe.TEST)

real5000 = caffe.Net('/u/vis/x1/redmunds/git/sound/environmental/realnodropout/sound_train2.prototxt',
                '/u/vis/x1/redmunds/git/sound/environmental/realnodropout_iter_5000.caffemodel',
                caffe.TEST)

real10000 = caffe.Net('/u/vis/x1/redmunds/git/sound/environmental/realnodropout/sound_train2.prototxt',
                '/u/vis/x1/redmunds/git/sound/environmental/realnodropout_iter_10000.caffemodel',
                caffe.TEST)

complex5000 = caffe.Net('/u/vis/x1/redmunds/git/sound/environmental/complexnet/sound_train2.prototxt',
                '/u/vis/x1/redmunds/git/sound/environmental/complexnet_iter_5000.caffemodel',
                caffe.TEST)

complex5000 = caffe.Net('/u/vis/x1/redmunds/git/sound/environmental/complexnet/sound_train2.prototxt',
                '/u/vis/x1/redmunds/git/sound/environmental/complexnet_iter_10000.caffemodel',
                caffe.TEST)



 
visualize_weights(net100, 'conv1', filename='conv1_100.png')
visualize_weights(net300, 'conv1', filename='conv1_300.png')
visualize_weights(net400, 'conv1', filename='conv1_400.png')
visualize_weights(net500, 'conv1', filename='conv1_500.png')
visualize_weights(real5000, 'conv1', filename='real5000.png')
visualize_weights(real10000, 'conv1', filename='real10000.png')
visualize_weights(complex5000, 'conv1', filename='complex5000.png')
visualize_weights(complex10000, 'conv1', filename='complex10000.png')