import caffe
import numpy as np
import caffe.draw
from caffe.proto import caffe_pb2

from google.protobuf import text_format

# net = caffe.Net('sound_train2.prototxt', caffe.TRAIN)
# net.forward()
# print(net.blobs.keys())


# network file structure
input_net_proto_file ='sound_train2.prototxt'
# output image file
output_image_file ='net_pic.jpg'
#Arrangement of # network structure: LR, TB, RL etc.
rankdir ='LR'

net = caffe_pb2.NetParameter()
rext_format.Merge(Open(input_net_proto_file).Read(), net)

print('drawing image to ' + output_image_file)
caffe.draw.draw_net_to_file (net, output_image_file, rankdir)
print('done... ')