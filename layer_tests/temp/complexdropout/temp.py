#forward pass test
import caffe

net = caffe.Net('temp.prototxt', caffe.TRAIN)

net.forward()

print 'dummy'
print net.blobs['data'].data.flatten()
print 'dropout'
print net.blobs['dropout'].data.flatten()
