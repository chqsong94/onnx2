import onnx as O
import os
import sys
import time
from collections import OrderedDict
from shutil import copyfile
import numpy as np
from IPython import embed
from single_layer import buildSingleLayerONNX
from multi_layer import buildMultiLayerONNX
import helper

import re
import json
import xlrd
import shutil


os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
# def get_config(ver, test_plan):
#     config = OrderedDict()

#     # elif ver == 17:
#     #     config['input_gen'] ='debug'
#     #     config['weight_gen'] ='debug'
#     #     config['bias_gen'] ='all0'
#     #     config['fn_excel'] = "test_cases_red_20181208_reorder.xlsx"
#     #     # config['row_numbers'] = list(range(97, 121))
#     #     config['sheet_name'] = 'v17_reshape_concat'
#     # else: raise "NotImplemented"

#     # assert (config['input_gen'] in ['random', 'debug', 'incre'])
#     # assert (config['weight_gen'] in ['random', 'debug', 'incre', 'all16'])
#     # assert (config['bias_gen'] in ['random', 'all0'])

#     config['fn_excel'] = test_plan
#     config['sheet_name'] = 'Sheet1' 
#     config['version'] = 'v{:03d}'.format(ver)
#     return config

def readTestCase(testCaseFileName, sheet_name):
    # Read content of test case table
    data = xlrd.open_workbook(testCaseFileName)
    table = data.sheet_by_name(sheet_name)

    cfgKey = table.row_values(0) ###############################
    # Normalize key values for tables
    for i in range(len(cfgKey)):
        key = cfgKey[i]
        cfgKey[i] = key.lstrip(" ").rstrip(" ").replace(" ", "_").lower()
    # Make configuration list
    cfgList = []
    for r in range(1, table.nrows):
    # for r in range(1, 2):
        cfgValue = table.row_values(r)
        cfgDict = dict(zip(cfgKey, cfgValue))
        cfgList.append(cfgDict)
    return cfgList


def getJson(cfgDict, model_name, idx="1"):
    mykeys = [
    "output_channel_num",
    "conv_pformat",	
    "conv_oformat",	
    "decompact",
    "decompress",	
    "conv_16b",	
    "acc",	
    "vstride2",	
    "hstride2",	
    "max",	
    "channel_start",	
    "channel_end",	
    "line",	
    "pconv_16b",	
    "gap_col_acc",	
    "pfunc_iformat",	
    "pfunc_oformat"]

    data1 = {"name": model_name} 
    data2 = {k: int(float(v)) for k, v in cfgDict.items() if k in mykeys}
    data = {"summary"+idx:{**data1, **data2}}
    return data


def genTestCase_single(cfgList,dir_output):
    if os.path.exists(dir_output):
        shutil.rmtree(dir_output)
    os.makedirs(dir_output)
    for cfgDict in cfgList:
        model_name = '{:05d}'.format(int(cfgDict["test_case_number"])) + '_' + \
        cfgDict["test_case_notes"].lstrip(" ").rstrip(" ").replace(" ", "_")   #changed

        model_name.lstrip(" ").rstrip(" ").replace(" ", "_")
        
        print('creating ' + model_name)
        my_jsonfile1 = getJson(cfgDict, model_name)
        my_jsonfile2, mysingleLayer = buildSingleLayerONNX(cfgDict, 1)
        model = helper.getModel(mysingleLayer)
        O.checker.check_model(model)

        value_info_shape_lst=[] # FOR this model
        for value_info in model.graph.value_info:
            for i in value_info.type.tensor_type.shape.dim:
                value_info_shape_lst.append(i.dim_value)
        if  any(i <= 0 for i in value_info_shape_lst) or (mysingleLayer.values_in == mysingleLayer.values_out):
            continue

        os.mkdir(dir_output + "/" + model_name)
        data = {**my_jsonfile1, **my_jsonfile2} 
        with open(dir_output+'/'+ model_name+'/'+'testcase.json', 'w') as outfile:  
            json.dump(data, outfile, indent=4)
        O.save(model, dir_output+'/'+ model_name + '/' + model_name +'.origin.onnx')


def genTestCase_multi(cfgList1, cfgList2,  dir_output):
    if os.path.exists(dir_output):
        shutil.rmtree(dir_output)
    os.makedirs(dir_output)
    for cfgDict1, cfgDict2 in zip(cfgList1, cfgList2):
        model_name = '{:05d}'.format(int(cfgDict1["test_case_number"])) + '_' + \
        cfgDict1["test_case_notes"].lstrip(" ").rstrip(" ").replace(" ", "_") + "_AND_" + \
        cfgDict2["test_case_notes"].lstrip(" ").rstrip(" ").replace(" ", "_")
        model_name.lstrip(" ").rstrip(" ").replace(" ", "_")
        print('creating ' + model_name)
        data1 = getJson(cfgDict1, model_name)
        data2 = getJson(cfgDict2, model_name, "2")
        my_jsonfile, mymultiLayer = buildMultiLayerONNX(cfgDict1, cfgDict2)
        model = helper.getModel(mymultiLayer)
        O.checker.check_model(model)
        value_info_shape_lst=[]
        for value_info in model.graph.value_info:
            for i in value_info.type.tensor_type.shape.dim:
                value_info_shape_lst.append(i.dim_value)
        if  any(i <= 0 for i in value_info_shape_lst) or (mymultiLayer.values_in == mymultiLayer.values_out):
            continue
        os.mkdir(dir_output + "/" + model_name)
        data = {**data1, **data2, **my_jsonfile}
        with open(dir_output+'/'+ model_name+'/'+'testcase.json', 'w') as outfile:  
            json.dump(data, outfile, indent=4)
        O.save(model, dir_output+'/'+ model_name + '/' + model_name +'.origin.onnx')



def gen_multi(excel_path, offset):
    output_path = "./gen"
    for version, test_plan in enumerate(os.listdir(excel_path)):
        # if test_plan == "multi":
        configs = OrderedDict()
        test_plan.lstrip(" ").rstrip(" ").replace(" ", "_")
        configs['fn_excel'] = test_plan
        configs['sheet_name'] = 'Sheet1' 
        configs['version'] = 'v2{:02d}'.format(version+offset)    #offset changed
        testCasesFileName = "{}/{}".format(excel_path, configs['fn_excel'])
        cfgList1 = readTestCase(testCasesFileName, sheet_name='layer1')
        cfgList2 = readTestCase(testCasesFileName, sheet_name='layer2')
        dir_output = "{}/gen_{}/{}_{}".format(output_path, excel_path.split("/")[-1], configs['version'],configs['fn_excel'].split('.')[0])
        genTestCase_multi(cfgList1, cfgList2, dir_output)


def gen_single(excel_path, offset):
    output_path = "./gen"
    for version, test_plan in enumerate(os.listdir(excel_path)):
        print(version, test_plan)
        if test_plan in ["elemSqaure_test_case.xlsx", "FCON_test_case.xlsx"]:
            continue
        configs = OrderedDict()
        test_plan.lstrip(" ").rstrip(" ").replace(" ", "_")
        configs['fn_excel'] = test_plan
        configs['sheet_name'] = 'Sheet1' 
        configs['version'] = 'v1{:02d}'.format(version+offset)   #offset changed
        testCasesFileName = "{}/{}".format(excel_path, configs['fn_excel'])
        cfgList = readTestCase(testCasesFileName, sheet_name=configs['sheet_name'])
        dir_output = "{}/gen_{}/{}_{}".format(output_path, excel_path.split("/")[-1], configs['version'], configs['fn_excel'].split('.')[0])
        genTestCase_single(cfgList, dir_output)



if __name__ == "__main__":
    np.random.seed(8)
    input_root = "./test_cases"
    gen_single(input_root + "/single_layer_test_case_8b", 1)
    gen_single(input_root +  "/single_layer_test_case_base_8b_deweight", 21)
    gen_single(input_root + "/single_layer_test_case_16b", 51)
    gen_single(input_root +  "/single_layer_test_case_base_16b_deweight", 71)
    
    gen_multi(input_root + "/multi_layer_test_case", 1)


