import tensorflow as tf
import numpy as np
import os
from tqdm import *


def load_pb(sess,pb_path,input_data_name,feature_maps_name):
    """
    提取输入和输出层
    """
    with tf.gfile.FastGFile(pb_path,"rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        sess.run(tf.global_variables_initializer())
        input_data = sess.graph.get_tensor_by_name(input_data_name+":0")
        feature_maps = [sess.graph.get_tensor_by_name(name+":0") for name in feature_maps_name]
        return input_data,feature_maps

def get_layers(sess, weight_ops):
    """
    获取核心操作层，用于重构cfg
    leakyrelu/maximum： 卷积层的激活函数
    concat/concatv2：routes层操作名
    biasadd：linear卷积层激活函数
    add：shortcut层
    resize：上采样层
    identity:yolo层
    """
    layers = []
    focus_layer = ["leakyrelu","concat","biasadd","add","resizenearestneighbor","identity","maximum","concatv2","resizebilinear"]
    for ops in weight_ops:
        layer_type = ops.type
        if layer_type.lower() in focus_layer:
            layer = sess.graph.get_tensor_by_name(ops.name+":0")
            if layer in layers:
                continue
            layers.append(layer)
            print(layer)
    return layers

def get_tensors_from_ops(name_scope="yolov3"):
    weight_ops = tf.get_default_graph().get_operations()
    #kernel_ops = [ops for ops in weight_ops if ops.name.split("/")[-1].lower() in ["conv2d","fusedbatchnorm","biasadd","identity"]]
    kernel_ops = [ops for ops in weight_ops if ops.type.lower() in ["conv2d","fusedbatchnorm","biasadd","identity"]]

    weight_ops = [ops for ops in weight_ops if "read" not in ops.name]
    layers = get_layers(sess, weight_ops)

    #获取所有需要保存的权重
    weight_ops = [ops for ops in weight_ops if ops.name.split("/")[-1] in ["kernel","gamma","beta","moving_mean","moving_variance","bias","weights","biases"]]
    tensors = [sess.graph.get_tensor_by_name(ops.name+":0") for ops in weight_ops]

    return layers, tensors

def load_img(img_path, img_size):
    img = cv2.imread(img_path, -1)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    float_img = np.asarray(img,np.float32)/255.
    return img,float_img

def get_pre_layers(layers,layer,pre_layers):
    op = tf.get_default_graph().get_operation_by_name(layer.op.name)
    inputs = op.inputs
    for item in inputs:
        if item in layers:
            if item not in pre_layers:
                pre_layers.append(item)
        else:
            get_pre_layers(layers,item,pre_layers)
    return

def get_conv_kernel(layer,kernels, count):
    if(count <= 0):
        return
    count -= 1
    op = tf.get_default_graph().get_operation_by_name(layer.op.name)
    inputs = op.inputs
    for item in inputs:
        if item.name.find("kernel") != -1 or item.name.find("weights") != -1:
            kernels.append(item)
        else:
            get_conv_kernel(item,kernels,count)
    return

def get_conv_param(net_layer):
    cur = net_layer['cur']
    pre = net_layer['pre']
    kernels = []
    get_conv_kernel(cur, kernels, 3)
    filters = kernels[0].shape[3]
    kernel_size = kernels[0].shape[1]

    if len(pre) == 0:
        strides = 1
    else:
        cur_size = cur.shape
        pre_size = pre[0].shape
        print(cur,pre_size[1],cur_size[1])
        strides = pre_size[1] // cur_size[1]
 
    return filters,kernel_size,strides

leaky_txt = \
"\
[convolutional]\n\
batch_normalize=1\n\
filters=%d\n\
size=%d\n\
stride=%d\n\
pad=1\n\
activation=leaky\n\n\
"
short_txt = \
"\
[shortcut]\n\
from=%d\n\
activation=linear\n\
"
linear_txt = \
"\
[convolutional]\n\
size=1\n\
stride=1\n\
pad=1\n\
filters=%d\n\
activation=linear\n\n\
"
yolov3_head = \
"\
[net]\n\
# Testing\n\
# batch=1\n\
# subdivisions=1\n\
# Training\n\
batch=64\n\
subdivisions=16\n\
width=608\n\
height=608\n\
channels=3\n\
momentum=0.9\n\
decay=0.0005\n\
angle=0\n\
saturation = 1.5\n\
exposure = 1.5\n\
hue=.1\n\
\n\
learning_rate=0.001\n\
burn_in=1000\n\
max_batches = 500200\n\
policy=steps\n\
steps=400000,450000\n\
scales=.1,.1\n\
"
yolo_features = ["6,7,8","3,4,5","0,1,2"]
yolo_layer_txt = \
"\
[yolo]\n\
mask = %s\n\
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326\n\
classes=%d\n\
num=9\n\
jitter=.3\n\
ignore_thresh = .7\n\
truth_thresh = 1\n\
random=1\n\
"


def get_shortcut_from(network,index):
    net_layer = network[index]
    pre_index = []
    for pre in net_layer['pre']:
        for i in range(len(network)):
            if network[i]['cur'] == pre:
                pre_index.append(i)
                break
    pre_index = [ (i-index) for i in pre_index]
    return min(pre_index)


def save_cfg(layers,cfg_path):
    f = open(cfg_path,"w")

    # 获取当前层的输入
    network = []
    for layer in layers:
        print(layer)
        net_layer = {}
        pre_layers = []
        get_pre_layers(layers,layer,pre_layers)
        net_layer['cur'] = layer
        net_layer['pre'] = pre_layers
        network.append(net_layer)

    feature_maps_index = 0
    for i in range(len(network)):
        net_layer = network[i]
        if len(net_layer['pre']) == 0:
            print(yolov3_head,file=f)
        elif len(net_layer['pre']) ==1:
            index = get_shortcut_from(network,i)
            if index != -1:
                print("[route]\nlayers=%d\n\n"%index,file=f)

        if net_layer['cur'].name.lower().find("leakyrelu") != -1:
            filters,kernel_size,strides = get_conv_param(net_layer)
            if strides == 2:
                print("#downsample\n\n",file=f)
            print(leaky_txt%(filters,kernel_size,strides) ,file=f)
        elif net_layer['cur'].name.lower().find("biasadd") != -1:
            filters = net_layer['cur'].shape[3]
            print(linear_txt %(filters),file=f)
        elif net_layer['cur'].name.lower().find("resize") != -1 or net_layer['cur'].name.lower().find("upsample") != -1:
            print("[upsample]\nstride=2\n",file=f)
        elif net_layer['cur'].name.lower().find("concat") != -1:
            index = get_shortcut_from(network,i)
            print("[route]\nlayers=-1,%d"%(index+i),file=f)
        elif net_layer['cur'].name.lower().find("add") != -1:
            route = get_shortcut_from(network,i)
            print(short_txt%(route),file=f)
        else:
            class_num = net_layer['cur'].shape[3]//3 - 5
            print(yolo_layer_txt%(yolo_features[feature_maps_index],class_num),file=f)
            feature_maps_index += 1

    f.close()
    return

def save_weights(sess, file_path, varlist):
    weight_info = np.array([0,2,0,0,0],dtype=np.int32)
    fp = open(file_path, "wb")
    weight_info.tofile(fp)
    for i in trange(len(varlist),desc="save weights"):
        length = 0
        layer_names = varlist[i].name.split("/")
        if layer_names[-1] != "kernel:0" and layer_names[-1] != "weights:0":
            continue
        next_layer_names = varlist[i+1].name.split("/")
        if next_layer_names[-1] == "bias:0" or next_layer_names[-1] == "biases:0":
            conv2d = varlist[i]
            bias = varlist[i+1]
            layer = [bias,conv2d]
        else:
            conv2d = varlist[i]
            #print(conv2d.name)
            gamma, beta, mean, var = varlist[i + 1:i + 5]
            layer = [beta,gamma,mean,var,conv2d]

        for param in layer:
            value = sess.run(param)
            if len(value.shape) == 4:
                value = np.transpose(value,(3,2,0,1)).reshape(-1)
            #print(value)
            value.tofile(fp)
        done = int((i+1)*100/len(varlist))
    fp.close()
    print("weights saved:", file_path)

if __name__ == '__main__':
    sess = tf.Session()

    # test: convert pb to darknet
    input_data_name = "input_data"
    feature_maps_name =["yolov3/yolov3_head/feature_map_1","yolov3/yolov3_head/feature_map_2","yolov3/yolov3_head/feature_map_3"]
    #加载pb模型，返回值未使用，可用于测试模型
    input_data,feature_maps = load_pb(sess, "./dianli_608_third_prune_20200303.pb",input_data_name, feature_maps_name)

    #根据name_scope提取网络的关键节点层和权重层
    layers, tensors = get_tensors_from_ops("yolov3")

    #根据关键节点的属性重构cfg
    save_cfg(layers,"./output.cfg")

    #保存权重层的值，即为weights
    save_weights(sess,"./output.weights",tensors)

