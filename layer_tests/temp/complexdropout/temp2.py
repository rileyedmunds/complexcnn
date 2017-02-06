#backward pass test
import caffe

net = caffe.Net('temp2.prototxt', caffe.TRAIN)

net.forward()
net.backward()

print 'dummy'
print net.blobs['data'].data.flatten()
print 'fc1'
print net.blobs['fc1'].data.flatten()
print 'dropout'
print net.blobs['dropout'].data.flatten()
print net.blobs['loss'].data

print 'dropout diff'
print net.blobs['dropout'].diff.flatten()
print 'fc1 diff'
print net.blobs['fc1'].diff.flatten()
