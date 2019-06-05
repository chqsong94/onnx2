import onnx as O
import os
import sys
import time
from collections import OrderedDict, namedtuple
import numpy as np
from helper import CreateConvOps, reluLayer_wrapper, poolingLayer_wrapper, GAPLayer_wrapper
import re
import json
import xlrd
from single_layer import buildSingleLayerONNX


def buildMultiLayerONNX(cfgDict1, cfgDict2):
    _, singleLayer1 = buildSingleLayerONNX(cfgDict1)

    _, singleLayer2 = buildSingleLayerONNX(cfgDict2, False)

    my_node_list = singleLayer1.node_list + singleLayer2.node_list
    my_values_in = singleLayer1.values_in + singleLayer2.values_in
    my_values_out= singleLayer1.values_out+ singleLayer2.values_out
    my_values_info=singleLayer1.values_info+singleLayer2.values_info
    MutiLayer = namedtuple("MultiLayer", ["node_list", "cfgDict", "values_in", "values_out" , "values_info"])

    muti_layer = MutiLayer(my_node_list, cfgDict2, my_values_in, my_values_out, my_values_info)
    return muti_layer





    # if conv_mode == 5 11 12 13 14 15 16  pfuncpool:
    #  outputchannel =    inputchannel
    # else:
    #     outputchannel = read from cfg


