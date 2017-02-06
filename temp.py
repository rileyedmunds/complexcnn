import caffe
import numpy as np

net = caffe.Net('sound_train2.prototxt', caffe.TRAIN)
net.forward()
print(net.blobs.keys())

for key, blob in net.blobs.items():
	print '{}: '.format(key),
	print 'shape={} '.format(blob.data.shape),
	print 'norm={} '.format(np.linalg.norm(blob.data))
	
 
