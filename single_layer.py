import onnx as O
import os
import sys
import time
from collections import OrderedDict
from shutil import copyfile
import numpy as np
from IPython import embed
from helper import CreateConvOps, reluLayer_wrapper, poolingLayer_wrapper, GAPLayer_wrapper
import re
import json
import xlrd


def buildSingleLayerONNX(cfgDict, needflatten=True):
    singleLayer = CreateConvOps(cfgDict)
    jsonFile = {}
    ################################################################################
    # check whether it is PFUNC or CONV type                                       #
    ################################################################################
    if cfgDict["conv/pfunc"] == "CONV":
        # this is in conv mode
        setting = cfgDict["conv_mode_in_hw_setting"]
        if setting == "0":
            # bypass
            testType = cfgDict["conv_mode_(in_model_operation)"]
            singleLayer.getCONVInputShape('conv')
            input_name_lst = []
            input_name_lst += singleLayer.construct_input('conv')
            if singleLayer.check_slice(): 
                convOpsName = singleLayer.construct_slice(testType, input_name_lst)
            else: convOpsName = input_name_lst
            singleLayer.getCONVOutputShape('bypass')

        elif setting in ["1", "6", "7"]:
            # conv 3x3 or ch4
            # having pad attr
            testType = cfgDict["conv_mode_(in_model_operation)"]
            singleLayer.getCONVInputShape('conv')
            input_name_lst = []
            input_name_lst += singleLayer.construct_input('conv')
            if singleLayer.check_slice(): 
                input_name_lst = singleLayer.construct_slice(testType, input_name_lst)
            singleLayer.getPaddingInfo()
            singleLayer.getCONVOutputShape('conv')
            input_name_lst += singleLayer.construct_weights(testType, 'conv')
            if int(cfgDict["bias_en"]):
                input_name_lst += singleLayer.construct_bias(testType, 'conv')
            convOpsName = singleLayer.construct_conv3x3(testType, input_name_lst) 


        elif setting in ['2', '3', '4', '5']:
            mydict = dict(zip(['2', '3', '4', '5'], [8, 4, 2, 1]))
            num_channels = mydict[setting]
       
        # elif testType == "2":
        #     # Group CONV3 by 8CH
        # elif testType == "3":
        #     # Group CONV3 by 4CH
        # elif testType == "4:
        #     # Group CONV3 by 2CH 
        # elif testType == "5":
        #     # conv 3x3 dw
        # having pad attr
            testType = cfgDict["conv_mode_(in_model_operation)"]
            singleLayer.getCONVInputShape('conv')
            input_name_lst = []
            input_name_lst += singleLayer.construct_input('conv')
            if singleLayer.check_slice(): 
                input_name_lst = singleLayer.construct_slice(testType, input_name_lst)
            singleLayer.getPaddingInfo()
            singleLayer.getCONVOutputShape('conv')
            input_name_lst += singleLayer.construct_weights(testType, 'group', num_channels)
            if int(cfgDict["bias_en"]):
                input_name_lst += singleLayer.construct_bias(testType, 'conv')
            convOpsName = singleLayer.construct_group_conv3x3(testType, input_name_lst, num_channels)
        
        elif setting == "8":
            # deconv 3x3 by2 only
            # having pad attr
            testType = cfgDict["conv_mode_(in_model_operation)"]
            singleLayer.getCONVInputShape('conv')
            input_name_lst = []
            input_name_lst += singleLayer.construct_input('conv')
            if singleLayer.check_slice(): 
                input_name_lst = singleLayer.construct_slice(testType, input_name_lst)
            singleLayer.getPaddingInfo()
            singleLayer.getCONVOutputShape('conv')
            input_name_lst += singleLayer.construct_weights(testType, 'conv')
            if int(cfgDict["bias_en"]):
                input_name_lst += singleLayer.construct_bias(testType, 'conv')
            convOpsName = singleLayer.construct_deconv(testType, input_name_lst)
        
        elif setting == "9":
            # dense
            # no pad attr
            testType = cfgDict["conv_mode_(in_model_operation)"]
            singleLayer.getCONVInputShape('conv')
            input_name_lst = []
            input_name_lst += singleLayer.construct_input('conv')
            # singleLayer.getPaddingInfo()
            if singleLayer.check_slice(): 
                input_name_lst = singleLayer.construct_slice(testType, input_name_lst)
            singleLayer.getCONVOutputShape('dense')
            input_name_lst += singleLayer.construct_weights(testType, 'dense')
            input_name_lst += singleLayer.construct_bias(testType, 'dense')
            convOpsName = singleLayer.construct_dense(testType, input_name_lst, needflatten)
        # elif testType == "10":
        #     # matrix
        elif setting == "11":
            # Elementwise product
            # no pad
            testType = cfgDict["conv_mode_(in_model_operation)"]
            singleLayer.getCONVInputShape('conv')
            input_name_lst = []
            input_name_lst += singleLayer.construct_input('em')
            # singleLayer.getPaddingInfo()
            if singleLayer.check_slice(): 
                input_name_lst = singleLayer.construct_slice(testType, input_name_lst)
            singleLayer.getPaddingInfo()
            singleLayer.getCONVOutputShape('conv')            
            convOpsName = singleLayer.construct_matmul(testType, input_name_lst)
        # elif testType == "12":
        #     # Elementwise square
        elif setting in ['13', '14']:
            # H-Upsample # V-Upsample
            testType = cfgDict["conv_mode_(in_model_operation)"]
            singleLayer.getCONVInputShape('conv')
            input_name_lst = []
            input_name_lst += singleLayer.construct_input('conv')
            # singleLayer.getPaddingInfo()
            if singleLayer.check_slice(): 
                input_name_lst = singleLayer.construct_slice(testType, input_name_lst)
            singleLayer.getPaddingInfo()
            singleLayer.getCONVOutputShape('conv')            
            convOpsName = singleLayer.construct_upsampling(testType, input_name_lst)
        
        elif setting == "15":
            # Conv 1x1 dw bn
            testType = cfgDict["conv_mode_(in_model_operation)"]
            singleLayer.getCONVInputShape('conv')
            input_name_lst = []
            input_name_lst += singleLayer.construct_input('conv')
            if singleLayer.check_slice(): 
                input_name_lst = singleLayer.construct_slice(testType, input_name_lst)
            singleLayer.getPaddingInfo()

            # print(singleLayer.kernel_size)
            # print
            if sum(singleLayer.kernel_size) == 2 and singleLayer.cfgDict['conv_stride'] == '1':
                singleLayer.getCONVOutputShape('bn')
                convOpsName = singleLayer.construct_bn(testType, input_name_lst)
                # print(convOpsName)
            else:
                singleLayer.getCONVOutputShape('conv') # depthwise 1x1
                input_name_lst += singleLayer.construct_weights(testType, 'group', 1)
                if int(cfgDict["bias_en"]):
                    input_name_lst += singleLayer.construct_bias(testType, 'conv')
                convOpsName = singleLayer.construct_group_conv3x3(testType, input_name_lst, 1)

        elif setting == "16":
            # Elementwise add
            testType = cfgDict["conv_mode_(in_model_operation)"]
            singleLayer.getCONVInputShape('conv')
            input_name_lst = []
            input_name_lst += singleLayer.construct_input('add')
            # singleLayer.getPaddingInfo()
            if singleLayer.check_slice(): 
                input_name_lst = singleLayer.construct_slice(testType, input_name_lst)
            singleLayer.getPaddingInfo()
            singleLayer.getCONVOutputShape('add')
            convOpsName = singleLayer.construct_add(testType, input_name_lst)
        else: raise "no such conv layer test mode"
    

        # sigmoid/tanh
        # input_bitwidth = 16 if int(cfgDict["relu_mode"]) in [3, 4] else 15
        output_bitwidth = 16 if int(singleLayer.cfgDict["relu_mode"]) in [3 , 4] else 15

        # default
        input_bitwidth = 15 if int(singleLayer.cfgDict["conv_16b"]) == 1 else 8
        output_bitwidth = output_bitwidth if int(singleLayer.cfgDict["pconv_16b"]) == 1 else 8


        jsonFile[testType] = {
            "input_bitwidth": input_bitwidth,
            "output_bitwidth": output_bitwidth
        }



    ################################################################################
    #         PCONV                                                                #
    ################################################################################
        if cfgDict["pconv_en"]:
            testType = "pconv"
            GAPLayer = GAPLayer_wrapper(cfgDict)
            reluLayer = reluLayer_wrapper(cfgDict)
            layerDict = OrderedDict()
            layerDict["GAP"] = GAPLayer if cfgDict["pconv_gap_en"] == '1' else None
            layerDict["Relu"] = reluLayer if cfgDict["pconv_relu_en"] == '1' else None
            input_str_info = convOpsName[0]
            input_shape = singleLayer.output_shape


            for c in layerDict:
                if layerDict[c] != None:
                    output = layerDict[c](testType+c, input_str_info, input_shape)
                    if output != None:
                        # below two line for next layer assign
                        input_str_info = output[3]
                        input_shape = output[2]
                        # below is for list concatenate
                        singleLayer.values_info += output[1] # output_value_info_lst = output[1]
                        singleLayer.node_list += output[0] # 	output_node_lst = output[0]

            if cfgDict["pconv_gap_en"] == "1" and cfgDict["pconv_relu_en"] == '1':
                # sigmoid/tanh
                gap_input_bitwidth = 16 if int(cfgDict["relu_mode"]) in [3, 4] else 15
                gap_output_bitwidth = 16 if int(cfgDict["relu_mode"]) in [3 , 4] else 15
                # default
                gap_input_bitwidth = gap_input_bitwidth if int(cfgDict["pconv_16b"]) == 1 else 8
                gap_output_bitwidth = gap_output_bitwidth if int(cfgDict["pconv_16b"]) == 1 else 8

                jsonFile[testType+"GAP"] = {
                    "input_bitwidth": gap_input_bitwidth,
                    "output_bitwidth": gap_output_bitwidth # it depends
                }

 
                jsonFile[testType+"Relu"] = {
                    "input_bitwidth": gap_output_bitwidth,
                    "output_bitwidth": 8 if int(singleLayer.cfgDict["conv_oformat"]) ==0 else 15 # it depends
                }

            elif cfgDict["pconv_gap_en"] == "1" and cfgDict["pconv_relu_en"] == '0':
                # sigmoid/tanh
                gap_input_bitwidth = 16 if int(cfgDict["relu_mode"]) in [3, 4] else 15
                gap_output_bitwidth = 16 if int(cfgDict["relu_mode"]) in [3 , 4] else 15
                # default
                gap_input_bitwidth = gap_input_bitwidth if int(cfgDict["pconv_16b"]) == 1 else 8
                gap_output_bitwidth = gap_output_bitwidth if int(cfgDict["pconv_16b"]) == 1 else 8

                jsonFile[testType+"GAP"] = {
                    "input_bitwidth": gap_input_bitwidth,
                    "output_bitwidth": gap_output_bitwidth # it depends
                }

            elif  cfgDict["pconv_relu_en"] == '1' and cfgDict["pconv_gap_en"] == "0":
                # sigmoid/tanh
                relu_input_bitwidth = 16 if int(cfgDict["relu_mode"]) in [3, 4] else 15
   
                # default
                relu_input_bitwidth = relu_input_bitwidth if int(cfgDict["pconv_16b"]) == 1 else 8
                relu_output_bitwidth = 15 if int(cfgDict["pconv_16b"]) == 1 else 8


                jsonFile[testType+"Relu"] = {
                    "input_bitwidth": relu_input_bitwidth,
                    "output_bitwidth": relu_output_bitwidth # it depends
                }




        else: print('pconv is not enable')
        singleLayer.values_out.append(singleLayer.values_info[-1])
        singleLayer.output_shape = input_shape

    ################################################################################
    ################################################################################
    # check whether it is PFUNC or CONV type                                       #
    ################################################################################
    elif cfgDict["conv/pfunc"] == "PFUNC":
        setting = cfgDict["conv_mode_in_hw_setting"]
        testType = cfgDict["conv_mode_(in_model_operation)"]
        if setting == "0":
            poolingLayer = poolingLayer_wrapper(cfgDict)
            singleLayer.getCONVInputShape('conv')
            input_name_lst = []
            convOpsName = singleLayer.construct_input('conv')
            if singleLayer.check_slice(): # and singleLayer.cfgDict["op_mode"] = '0' and singleLayer.cfgDict["pool_mode"] != '0':
                if singleLayer.cfgDict["op"] == '0' and singleLayer.cfgDict["pool_mode"] == '0':
                    pass
                else:
                    convOpsName = singleLayer.construct_slice(testType, convOpsName)
            output = poolingLayer(testType, convOpsName[0], singleLayer.input_shape)
            singleLayer.values_info += output[1] # output_value_info_lst = output[1]
            singleLayer.node_list += output[0] # 	output_node_lst = output[0]

        singleLayer.values_out.append(singleLayer.values_info[-1])
        singleLayer.output_shape = output[2]


        bitmode = dict(zip(['0', '1', '2', '3'], [8, 15, 8, 16]))
        jsonFile[testType] = {
            "input_bitwidth": bitmode[ singleLayer.cfgDict["pfunc_iformat"] ],
            "output_bitwidth": bitmode[ singleLayer.cfgDict["pfunc_oformat"]]
        }
    else: raise "no such single layer test"
    return jsonFile, singleLayer

