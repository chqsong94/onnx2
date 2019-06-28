import logging
import numpy as np
import onnx as O
from onnx import TensorProto


# def getPadding(size, kernel_size, strides):
# 	if size[0] % strides[0] == 0:
# 		pad_h = max(kernel_size[0] - strides[0], 0)
# 	else:
# 		pad_h = max(kernel_size[0] - (size[0] % strides[0]), 0)
# 	if size[1] % strides[1] == 0:
# 		pad_w = max(kernel_size[1] - strides[1], 0)
# 	else:
# 		pad_w = max(kernel_size[1] - (size[1] % strides[1]), 0)
# 	return [pad_h//2, pad_w//2, pad_h-pad_h//2, pad_w-pad_w//2]


class CreateConvOps():
	def __init__(self, cfgDict):
		self.node_list = []
		self.values_in = []
		self.values_out = []
		self.values_info = []
		self.cfgDict = cfgDict

		self.col1 = int(float(self.cfgDict['col_st']))
		self.row1 = int(float(self.cfgDict['row_st']))

		self.col2 = int(self.cfgDict["input_size_w_(col)"]) if int(float(self.cfgDict['col_out']))==0 else self.col1 + int(float(self.cfgDict['col_out']))
		self.row2 = int(self.cfgDict["input_size_h_(row)"]) if int(float(self.cfgDict['row_out']))==0 else self.row1 + int(float(self.cfgDict['row_out']))
		

		self.kernel_size = (int(self.cfgDict["kernel_size_h"]), int(self.cfgDict["kernel_size_w"]))
	
	def check_slice(self):
		if (self.col1 + self.row1 + int(float(self.cfgDict['col_out'])) + int(float(self.cfgDict['row_out'])) ) == 0:
			flag = False
		else: flag = True
		return flag


	def getPaddingInfo(self):
		self.paddingTop = int(self.cfgDict['up_padding_t'])
		self.paddingBottom = int(self.cfgDict['dn_padding_b'])
		self.paddingLeft = int(self.cfgDict['left_padding_l'])
		self.paddingRight = int(self.cfgDict['right_padding_r'])
		self.padding_info = [ self.paddingTop, self.paddingLeft, self.paddingBottom, self.paddingRight]
	
	
	def getCONVInputShape(self, mode):
		"""
		this is the conv part input shape
		"""
		self.input_shape = (1, int(self.cfgDict["input_channel_num"]), int(self.cfgDict["input_size_h_(row)"]), int(self.cfgDict["input_size_w_(col)"]))
	

	def construct_input(self, mode, layeridx=None):
		if mode in ['add', 'em']:
			input1 = O.helper.make_tensor_value_info('Input1'+str(layeridx), O.TensorProto.FLOAT, list(self.input_shape))
			self.values_info.append(input1)
			input2 = O.helper.make_tensor_value_info('Input2'+str(layeridx), O.TensorProto.FLOAT, list(self.input_shape))
			self.values_info.append(input2)  
			self.values_in += [input1, input2]
			output_name = ['Input1'+str(layeridx),'Input2'+str(layeridx)]  
		else:
			inputs = O.helper.make_tensor_value_info('Input', O.TensorProto.FLOAT, list(self.input_shape))
			self.values_info.append(inputs) 
			self.values_in += [inputs]	
			output_name = ['Input'] 
		return output_name #for next layer


	def construct_slice(self, testType, input_name_lst):
		# print(input_name_lst)
		self.input_shape = (1, self.input_shape[1], self.row2-self.row1, self.col2-self.col1)
		if len(input_name_lst)==1:
			slice_node = O.helper.make_node(
				op_type = 'Slice',
				inputs = input_name_lst,
				outputs = ['slice_out' + testType],
				starts= np.array([0, 0, self.row1, self.col1], dtype=np.int64),
				ends= np.array([1, self.input_shape[1], self.row2, self.col2], dtype=np.int64),
				name=testType+"slice"
				)
	
			self.node_list.append(slice_node)
			output = O.helper.make_tensor_value_info('slice_out'+testType, O.TensorProto.FLOAT, list(self.input_shape))
			self.values_info.append(output)	
			output_name = ['slice_out' + testType]

		else: # has two inputs
			slice_node1 = O.helper.make_node(
				op_type = 'Slice',
				inputs = [input_name_lst[0]],
				outputs = ['slice_out1' + testType],
				starts= np.array([0, 0, self.row1, self.col1], dtype=np.int64),
				ends= np.array([0, self.input_shape[1], self.row2, self.col2], dtype=np.int64),
				name=testType+"slice1"
				)

			self.node_list.append(slice_node1)
			output1 = O.helper.make_tensor_value_info('slice_out1'+testType, O.TensorProto.FLOAT, list(self.input_shape))
			self.values_info.append(output1)

			slice_node2 = O.helper.make_node(
				op_type = 'Slice',
				inputs = [input_name_lst[1]],
				outputs = ['slice_out2' + testType],
				starts= np.array([0, 0, self.row1, self.col1], dtype=np.int64),
				ends= np.array([0, self.input_shape[1], self.row2, self.col2], dtype=np.int64),
				name= testType+"slice2"
				)
			self.node_list.append(slice_node2)
			output2 = O.helper.make_tensor_value_info('slice_out2'+testType, O.TensorProto.FLOAT, list(self.input_shape))
			self.values_info.append(output2)
			output_name = ['slice_out1'+testType, 'slice_out2'+testType]
		return output_name


	def getCONVOutputShape(self, mode):
		"""
		this is the conv part output shape
		"""
		self.num_channels = int(self.cfgDict['channel_end']) - int(self.cfgDict['channel_start']) + 1

		# if int(self.cfgDict["conv_mode_in_hw_setting"]) in [5, 11, 12, 13, 14, 15, 16] and self.cfgDict["conv/pfunc"] == "CONV":
		# 	self.num_channels = self.input_shape[1]

		if mode == 'dense':		
			self.output_shape = (1,  self.num_channels, 1, 1)

		elif mode in ['add', 'bypass', 'bn']:
			self.output_shape = self.input_shape
			self.kernel_size = (int(self.cfgDict["kernel_size_h"]), int(self.cfgDict["kernel_size_w"]))
		elif mode == 'conv':	
			self.kernel_size = (int(self.cfgDict["kernel_size_h"]), int(self.cfgDict["kernel_size_w"]))
			strides = (int(self.cfgDict["conv_stride"]), int(self.cfgDict["conv_stride"]))
			output_rows = int((self.input_shape[2] - self.kernel_size[0] + self.paddingTop + self.paddingBottom)/strides[0] + 1)
			output_cols = int((self.input_shape[3] - self.kernel_size[1] + self.paddingRight + self.paddingLeft)/strides[1] + 1)
			self.output_shape = (1,  self.num_channels, output_rows, output_cols)
		elif mode == 'deconv':
			self.kernel_size = (int(self.cfgDict["kernel_size_h"]), int(self.cfgDict["kernel_size_w"]))
			strides = (int(self.cfgDict["conv_stride"]), int(self.cfgDict["conv_stride"]))
			output_rows = int((self.input_shape[2] - 1) * strides[0] + self.kernel_size[0])
			output_cols = int((self.input_shape[3] - 1) * strides[1] + self.kernel_size[1])
			self.output_shape = (1,  self.num_channels, output_rows, output_cols)

		# strides = (2, 2)
		# expanded_row = (self.input_shape[2] - 1) * (strides[0] -1) + self.input_shape[2]
		# expanded_col = (self.input_shape[3] - 1) * (strides[0] -1) + self.input_shape[3]

		else: raise "no such mode"


	def construct_weights(self, testType, mode, channelsPerGroup =1):
		if mode == 'dense':
			weights_shape = [self.num_channels, self.input_shape[1]*self.input_shape[2]*self.input_shape[3]]
			weights_value = np.ones(shape=weights_shape).ravel()
		elif mode == 'conv': #conv
			weights_shape = [self.output_shape[1], self.input_shape[1], self.kernel_size[0], self.kernel_size[1]]
			if self.kernel_size[0]*self.kernel_size[1] == 1:
				a = np.ones(self.kernel_size[0]*self.kernel_size[1]).reshape(self.kernel_size[0],self.kernel_size[1])
			else :
				a = np.arange(self.kernel_size[0]*self.kernel_size[1]).reshape(self.kernel_size[0],self.kernel_size[1])
			a = np.repeat(a[:, :, np.newaxis], self.input_shape[1], axis=2)
			a = np.repeat(a[:, :, :, np.newaxis], self.output_shape[1], axis=3)
			weights_value = a.ravel()

		elif mode == 'group':
			weights_shape = [self.output_shape[1], channelsPerGroup,  self.kernel_size[0], self.kernel_size[1]]
			if self.kernel_size[0]*self.kernel_size[1] == 1:
				a = np.ones(self.kernel_size[0]*self.kernel_size[1]).reshape(self.kernel_size[0],self.kernel_size[1])
			else :
				a = np.arange(self.kernel_size[0]*self.kernel_size[1]).reshape(self.kernel_size[0],self.kernel_size[1])
			a = np.repeat(a[:, :, np.newaxis], channelsPerGroup, axis=2)
			a = np.repeat(a[:, :, :, np.newaxis], self.output_shape[1], axis=3)
			weights_value = a.ravel()


		elif mode == 'deconv': #conv
			weights_shape = [self.input_shape[1], self.output_shape[1], self.kernel_size[0], self.kernel_size[1]]
			if self.kernel_size[0]*self.kernel_size[1] == 1:
				a = np.ones(self.kernel_size[0]*self.kernel_size[1]).reshape(self.kernel_size[0],self.kernel_size[1])
			else :
				a = np.arange(self.kernel_size[0]*self.kernel_size[1]).reshape(self.kernel_size[0],self.kernel_size[1])
			a = np.repeat(a[:, :, np.newaxis], self.input_shape[1], axis=2)
			a = np.repeat(a[:, :, :, np.newaxis], self.output_shape[1], axis=3)
			weights_value = a.ravel()

		else: raise "no such mode"

		
		weights_value = np.random.normal(size=weights_shape).ravel()
		w_info =  O.helper.make_tensor_value_info('weights'+testType, O.TensorProto.FLOAT, list(weights_shape))
		weights_tensor = O.helper.make_tensor('weights_tensor', O.TensorProto.FLOAT, weights_shape, weights_value)
		w_node = O.helper.make_node(
		op_type = "Constant",
		inputs = [],
		outputs = ['weights' + testType],
		name='weights_1',
		value=weights_tensor
		)

		self.node_list.append(w_node)
		self.values_info.append(w_info)
		output_name = ['weights'+testType]
		return output_name


	def construct_bias(self, testType, mode):
		if mode == 'dense':
			bias_shape = [1, self.output_shape[1]]
			bias_value = np.random.normal(size = bias_shape) if int(self.cfgDict["bias_en"]) else np.zeros(bias_shape)
		else: #mode == 'conv':
			bias_shape = [self.output_shape[1]]
			bias_value = np.random.normal(size= bias_shape)

		b_info = O.helper.make_tensor_value_info('bias'+testType, O.TensorProto.FLOAT, bias_shape)
		bias_tensor = O.helper.make_tensor('bias_tensor',O.TensorProto.FLOAT,bias_shape, bias_value.ravel())
		b_node = O.helper.make_node(
		op_type = "Constant",
		inputs = [],
		outputs = ['bias' + testType],
		name='bias_1',
		value=bias_tensor,
		)
		self.node_list.append(b_node)
		self.values_info.append(b_info)
		output_name = ['bias'+testType]
		return output_name
		

	def construct_pad(self, testType, input_name_lst):
		# this is stand alone pad
		pad_node = O.helper.make_node(
		op_type = 'Pad', # node name
		inputs = input_name_lst, # inputs
		outputs = [testType+'padding'], # outputs
		#mode='constant', # Attributes
		name=testType + 'pad',
		pads=[0,0, self.padding_info[0], self.padding_info[1], 0, 0, self.padding_info[2], self.padding_info[3]]
		)
		self.node_list.append(pad_node)

		after_pad_col = int(self.input_shape[2] + self.paddingTop + self.paddingBottom)
		after_pad_row = int(self.input_shape[3] + self.paddingLeft + self.paddingRight)

		self.input_shape = [self.input_shape[0], self.input_shape[1], after_pad_row, after_pad_col]

		pad_info = O.helper.make_tensor_value_info(testType+'padding', O.TensorProto.FLOAT, self.input_shape)
		self.values_info.append(pad_info)
		output_name = [testType+'padding']
		return output_name
	

	def construct_conv3x3(self, testType, input_name_lst):
		kernel_size = (int(self.cfgDict["kernel_size_h"]), int(self.cfgDict["kernel_size_w"]))
		strides = (int(self.cfgDict["conv_stride"]), int(self.cfgDict["conv_stride"]))
		conv_node = O.helper.make_node(
		op_type = 'Conv',
		inputs = input_name_lst,
		outputs = [testType +'_out'],
		name = str(testType),
		group = 1,
		kernel_shape=list(kernel_size),
		pads=self.padding_info,
		strides=list(strides),
		)
		self.node_list.append(conv_node)
		output = O.helper.make_tensor_value_info(testType+'_out', O.TensorProto.FLOAT, list(self.output_shape))
		self.values_info.append(output)	
		output_name = [testType +'_out']
		return output_name


	def construct_group_conv3x3(self, testType, input_name_lst, channelsPerGroup):
		kernel_size = (int(self.cfgDict["kernel_size_h"]), int(self.cfgDict["kernel_size_w"]))
		strides = (int(self.cfgDict["conv_stride"]), int(self.cfgDict["conv_stride"]))	
		group_conv_node = O.helper.make_node(
		op_type = 'Conv',
		inputs = input_name_lst,
		outputs = [testType +'_out'],
		name = str(testType),
		group = int(self.input_shape[1] / channelsPerGroup),
		kernel_shape=list(kernel_size),
		pads=self.padding_info,
		strides=list(strides),
		)
		self.node_list.append(group_conv_node)
		output = O.helper.make_tensor_value_info(testType+'_out', O.TensorProto.FLOAT, list(self.output_shape))
		self.values_info.append(output)	
		output_name = [testType +'_out']
		return output_name


	def construct_deconv(self, testType, input_name):
		kernel_size = (int(self.cfgDict["kernel_size_h"]), int(self.cfgDict["kernel_size_w"]))
		strides = (int(self.cfgDict["conv_stride"]), int(self.cfgDict["conv_stride"]))		
		deconv_node = O.helper.make_node(
			op_type = 'ConvTranspose',
			inputs = input_name,
			outputs = [testType +'_out'],
			name = str(testType),
			group = 1,
			kernel_shape=list(kernel_size),
			pads=[self.kernel_size[0]-1, self.kernel_size[1]-1, self.kernel_size[0]-1, self.kernel_size[1]-1],
			strides=list(strides),
			)
		self.node_list.append(deconv_node)
		output = O.helper.make_tensor_value_info(testType+'_out', O.TensorProto.FLOAT, list(self.output_shape))
		self.values_info.append(output)	
		output_name = [testType +'_out']
		return output_name


	def construct_dense(self, testType, input_name, layeridx):
		if layeridx == 1:
			flattened_output_shape = (self.input_shape[0], self.input_shape[1]*self.input_shape[2]*self.input_shape[3], 1, 1) # this is intermediate shape
			output = O.helper.make_tensor_value_info('flatten'+testType, O.TensorProto.FLOAT, list(flattened_output_shape))
			flatten_node = O.helper.make_node(
				op_type = 'Flatten',
				inputs=[input_name[0]],
				outputs=['flatten'+testType],
				name='flatten_1',
				axis=1)
			self.node_list.append(flatten_node)
			self.values_info.append(output)

			dense_node = O.helper.make_node(
				op_type = 'Gemm',
				inputs = ['flatten'+testType]+ input_name[1:],
				outputs = [testType+'_out'],
				name=testType,
				alpha=1.0,
				beta=1.0,
				transA=0,
				transB=0
				)
			self.node_list.append(dense_node)
			output = O.helper.make_tensor_value_info(testType+'_out', O.TensorProto.FLOAT, list(self.output_shape))

		else:
			dense_node = O.helper.make_node(
				op_type = 'Gemm',
				inputs = input_name,
				outputs = [testType+'_out'],
				name=testType,
				alpha=1.0,
				beta=1.0,
				transA=0,
				transB=0
				)
			self.node_list.append(dense_node)
			output = O.helper.make_tensor_value_info(testType+'_out', O.TensorProto.FLOAT, list(self.output_shape))
		self.values_info.append(output)	
		output_name = [testType +'_out']
		return output_name


	def construct_add(self, testType, input_name):
		# input1 = self.construct_pad([input_name[0]])
		# input2 = self.construct_pad([input_name[1]])
		add_node = O.helper.make_node(
			op_type = 'Add',
			inputs = input_name,
			outputs = [testType+'_out'],
			name=str(testType),
			)
		self.node_list.append(add_node)
		output = O.helper.make_tensor_value_info(testType+'_out', O.TensorProto.FLOAT, list(self.output_shape))
		self.values_info.append(output)	
		output_name = [testType +'_out']
		return output_name


	def construct_bn(self, testType, input_str_info):
		# node_lst = []
		# value_info_lst = []
		# input_str_info = self.construct_pad(input_str_info)
		input_bn_shape = self.input_shape
		# gamma
		scale_bn_info = O.helper.make_tensor_value_info('scale_bn_info'+testType, O.TensorProto.FLOAT, [input_bn_shape[1]])
		scale_bn_tensor = O.helper.make_tensor('scale_bn_tensor', O.TensorProto.FLOAT, 
		[input_bn_shape[1]], np.random.normal(size =input_bn_shape[1]))
		node_scale_bn = O.helper.make_node( 
		op_type='Constant',
		inputs=[],
		outputs=['scale_bn_info'+testType],
		name='Scale_BN',
		value=scale_bn_tensor
		)
		self.node_list.append(node_scale_bn)
		self.values_info.append(scale_bn_info)

		# beta
		bias_bn_info = O.helper.make_tensor_value_info('bias_bn_info'+testType, O.TensorProto.FLOAT, [input_bn_shape[1]])
		bias_bn_tensor = O.helper.make_tensor('bias_bn_tensor', O.TensorProto.FLOAT, 
		[input_bn_shape[1]], np.random.normal(size =input_bn_shape[1]))
		node_bias_bn = O.helper.make_node( 
		op_type='Constant',
		inputs=[],
		outputs=['bias_bn_info'+testType],
		name='Bias_BN',
		value=bias_bn_tensor
		)
		self.node_list.append(node_bias_bn)
		self.values_info.append(bias_bn_info)

		# mean
		mean_bn_info = O.helper.make_tensor_value_info('mean_bn_info'+testType, O.TensorProto.FLOAT, [input_bn_shape[1]])
		mean_bn_tensor = O.helper.make_tensor('mean_bn_tensor', O.TensorProto.FLOAT, 
		[input_bn_shape[1]], np.random.normal(size =input_bn_shape[1]))
		node_mean_bn = O.helper.make_node( 
		op_type='Constant',
		inputs=[],
		outputs=['mean_bn_info'+testType],
		name='Mean_BN',
		value=mean_bn_tensor
		)
		self.node_list.append(node_mean_bn)
		self.values_info.append(mean_bn_info)

		# var
		var_bn_info = O.helper.make_tensor_value_info('var_bn_info'+testType, O.TensorProto.FLOAT, [input_bn_shape[1]])
		var_bn_tensor = O.helper.make_tensor('var_bn_tensor', O.TensorProto.FLOAT, 
		[input_bn_shape[1]], abs(np.random.normal(size =input_bn_shape[1])))
		node_var_bn = O.helper.make_node( 
		op_type='Constant',
		inputs=[],
		outputs=['var_bn_info'+testType],
		name='Var_BN',
		value=var_bn_tensor
		)
		self.node_list.append(node_var_bn)
		self.values_info.append(var_bn_info)
		output_bn_info = O.helper.make_tensor_value_info('output_bn_info'+testType, O.TensorProto.FLOAT, list(self.output_shape))
		node_bn = O.helper.make_node( 
		op_type='BatchNormalization',
		inputs=input_str_info + ['scale_bn_info'+testType, 'bias_bn_info'+testType, 'mean_bn_info'+testType, 'var_bn_info'+testType],
		outputs=['output_bn_info'+testType],
		name=testType,
		epsilon=1e-05,
		momentum=0.9,
		# spatial=1,
		# is_test=1
		)
		self.node_list.append(node_bn)
		self.values_info.append(output_bn_info)
		output_name = ['output_bn_info'+testType]
		return output_name
		

	def construct_matmul(self, testType, input_name):
		# input1 = self.construct_pad([input_name[0]])
		# input2 = self.construct_pad([input_name[1]])
		matmul_node = O.helper.make_node(
			op_type = 'Mul',
			inputs = input_name,
			outputs = [testType+'_out'],
			name=testType,
			)
		self.node_list.append(matmul_node)
		output = O.helper.make_tensor_value_info(testType+'_out', O.TensorProto.FLOAT, list(self.output_shape))
		self.values_info.append(output)	
		output_name = [testType +'_out']
		return output_name


	def construct_upsampling(self, testType, input_name):
		v_scale = float(self.cfgDict['vus_scale'])
		h_scale = float(self.cfgDict['hus_scale'])
		self.output_shape = list(self.output_shape)
		self.output_shape[2] = int(self.output_shape[2] * v_scale)
		self.output_shape[3] = int(self.output_shape[3] * h_scale)
		self.output_shape = tuple(self.output_shape)
		output = O.helper.make_tensor_value_info(testType+'_out', O.TensorProto.FLOAT, list(self.output_shape))
		up_node = O.helper.make_node(
			op_type = 'Upsample',
			inputs = input_name,
			outputs = [testType+'_out'],
			name=testType,
			mode= 'nearest' if self.cfgDict["near"] else 'bilinear',
			scales = [1.0, 1.0, v_scale, h_scale],
			)

		self.node_list.append(up_node)
		self.values_info.append(output)	
		output_name = [testType +'_out']
		return output_name


################################################################################
#			this is for pconv												   #	
################################################################################
def reluLayer_wrapper(cfgDict):
	def reluLayer(testType, input_str_info, input_relu_shape):
		relu_mode = int(cfgDict["relu_mode"])
		relu6_clamp = cfgDict["relu6_clamp"]
		node_lst = []
		value_info_lst = []
		output_relu_shape = input_relu_shape
		output_relu_info = O.helper.make_tensor_value_info('output_relu_info'+testType, O.TensorProto.FLOAT, 
		list(output_relu_shape))
		
		if relu_mode == 0:
			node_relu = O.helper.make_node(
			op_type='Relu',
			inputs = [input_str_info],
			outputs = ['output_relu_info'+testType],
			name = testType
			)			
		elif relu_mode == 2:
			slope_relu_info = O.helper.make_tensor_value_info('slope_relu_info'+testType, O.TensorProto.FLOAT, list(input_relu_shape[1:]))
			slope_relu_tensor = O.helper.make_tensor('slope_relu_tensor', O.TensorProto.FLOAT, 
				list(input_relu_shape[1:]), np.random.normal(size =input_relu_shape[1]))
			node_slope = O.helper.make_node( 
			op_type='Constant',
			inputs=[],
			outputs=['slope_relu_info'+testType],
			name=testType,
			value=slope_relu_tensor
			)
			node_lst.append(node_slope)
			value_info_lst.append(slope_relu_info)
			# channel, heigh and width, but want to share axis in w and h
			node_relu = O.helper.make_node(
			op_type='PRelu',
			inputs = [input_str_info, 'slope_relu_info'+testType],
			outputs = ['output_relu_info'+testType],
			name = testType
			)
		elif relu_mode == 3:
			node_relu = O.helper.make_node(
			'Clip',
			inputs=[input_str_info],
			outputs=['output_relu_info'+testType],
			min=0.0,
			max= 6.0 if relu6_clamp != "MAX" else 8.0,
			name = testType
			)
		elif relu_mode == 4:
			node_relu = O.helper.make_node(
			'Sigmoid',
			inputs=[input_str_info],
			outputs=['output_relu_info'+testType],
			name = testType
			)
		elif relu_mode == 5:
			node_relu = O.helper.make_node(
			'Tanh',
			inputs=[input_str_info],
			outputs=['output_relu_info'+testType],
			name = testType
			)
		elif relu_mode == 1:
			node_relu = O.helper.make_node(
			op_type = 'LeakyRelu',
			inputs=[input_str_info],
			outputs=['output_relu_info'+testType],
			alpha=0.1,
			name = testType
			)
		else:
			results = None
			return results
		node_lst.append(node_relu)
		value_info_lst.append(output_relu_info)  
		results = [node_lst, value_info_lst, output_relu_shape, 'output_relu_info'+testType]
		return results
	return reluLayer


def GAPLayer_wrapper(cfgDict):
	def GAPLayer(testType, input_str_info, input_pool_shape):
		node_lst = []
		value_info_lst = []
		output_toflatten_shape = (input_pool_shape[0], input_pool_shape[1], 1, 1) # output size of gap
		output_toflatten_info = O.helper.make_tensor_value_info('output_toflatten_info'+testType, O.TensorProto.FLOAT, list(output_toflatten_shape))
		gap_node = O.helper.make_node(
		op_type='GlobalAveragePool',
		inputs=[input_str_info],
		outputs=['output_toflatten_info'+testType],
		name=testType,
		# kernel_shape=[int(pool_size), int(pool_size)],
		# pads=pads,
		# strides=strides
		)
		node_lst.append(gap_node)
		value_info_lst.append(output_toflatten_info)


		# output_pool_shape = (input_pool_shape[0], input_pool_shape[1])
		# output_pool_info = O.helper.make_tensor_value_info('output_pool_info'+testType, O.TensorProto.FLOAT, list(output_pool_shape))
		# node_pool = O.helper.make_node(
		# 'Flatten',
		# inputs=['output_toflatten_info'+testType],
		# outputs=['output_pool_info'+testType],
		# name='flatten',
		# axis=1)
		# node_lst.append(node_pool)
		# value_info_lst.append(output_pool_info)
		# results = [node_lst, value_info_lst, output_pool_shape,'output_pool_info'+testType]
		results = [node_lst, value_info_lst, output_toflatten_shape,'output_toflatten_info'+testType]


		return results
	return GAPLayer


def poolingLayer_wrapper(cfgDict):
	pool_dict = dict(zip( ['0', '1', '2', '3', '4', '5' ], [(0,0), (2,1), (2,2), (3,1), (3,2), (3,3)])) #sz and stride
	def poolingLayer(testType, input_str_info, input_pool_shape):
		node_lst = []
		value_info_lst = []
		#  if np.sum(padding_info) > 0
		paddingTop = int(cfgDict['pool_padding_up'])
		paddingBottom = int(cfgDict['pool_padding_dn'])
		paddingLeft = int(cfgDict['pool_padding_left'])
		paddingRight = int(cfgDict['pool_padding_right'])
		padding_info = [ paddingTop, paddingLeft, paddingBottom, paddingRight]

		op_mode = cfgDict["op"]
		pool_mode = cfgDict["pool_mode"]
		pool_size, pool_stride = pool_dict[pool_mode]
		
		if op_mode == '0':
			# max pool
			if pool_mode == '0':
				# roi pooling
				x1 = int(cfgDict['col_st'])
				y1 = int(cfgDict['row_st'])
				x2 = x1 + int(cfgDict['col_out'])
				y2 = y1 + int(cfgDict['row_out'])

				# if (x1 + x2) ==0:
				# 	if (y1 + y2) == 0:
				# 		loc = np.array([ [0, 0, 0, input_pool_shape[3], input_pool_shape[2]] ])
				# 	else:
				# 		loc = np.array([ [0, 0, y1, input_pool_shape[3], y2] ])
				# else:
				# 	if (y1 + y2) == 0:
				# 		loc = np.array([ [0, x1, 0, x2, input_pool_shape[2]] ])
				# 	else:
				# 		loc = np.array([[0, x1, y1, x2, y2]])

				if int(cfgDict['col_out']) ==0 and int(cfgDict['row_out']) !=0:
					loc = np.array([ [0, x1, y1, input_pool_shape[3], y2] ])
				elif int(cfgDict['row_out']) ==0 and int(cfgDict['col_out']) !=0:
					loc = np.array([ [0, x1, y1, x2, input_pool_shape[2]] ])
				elif int(cfgDict['col_out'])==0 and int(cfgDict['row_out']) ==0:
					loc = np.array([ [0, x1, y1, input_pool_shape[3], input_pool_shape[2]] ])
				else:
					loc = np.array([[0, x1, y1, x2, y2]])
				
				rois_info = O.helper.make_tensor_value_info('rois_info'+testType, O.TensorProto.FLOAT, [1, 5])
				rois_tensor = O.helper.make_tensor('rois_tensor', O.TensorProto.FLOAT, [1, 5], loc.ravel())
				rois_node = O.helper.make_node( 
				op_type='Constant',
				inputs=[],
				outputs=['rois_info'+testType],
				name='rois',
				value=rois_tensor
				)
				node_lst.append(rois_node)
				value_info_lst.append(rois_info)

				# now calculate pooled shape
			
				if int(cfgDict["roi_pooling_row_cnt"]) ==0:
					height = 1
				else:
					height = int(cfgDict["roi_pooling_row_cnt"]) if int(cfgDict["input_size_h_(row)"]) % int(cfgDict["roi_pooling_row_cnt"]) == 0 else int(cfgDict["roi_pooling_row_cnt"]) + 1
				if int(cfgDict["roi_pooling_col_cnt"]) ==0:
					width = 1
				else:
					width = int(cfgDict["roi_pooling_col_cnt"]) if int(cfgDict["input_size_w_(col)"]) % int(cfgDict["roi_pooling_col_cnt"]) == 0 else int(cfgDict["roi_pooling_col_cnt"]) + 1
				output_pool_shape = (input_pool_shape[0], input_pool_shape[1], height, width)
				output_pool_info = O.helper.make_tensor_value_info('output_pool_info'+testType, O.TensorProto.FLOAT, list(output_pool_shape))

				pool_node = O.helper.make_node(
				op_type='MaxRoiPool',
				inputs=[input_str_info, 'rois_info'+testType],
				outputs=['output_pool_info'+testType],
				name=testType,
				pooled_shape=(height, width),
				spatial_scale= 1.0,
				)


			else:
				result_row = int((input_pool_shape[2] - int(pool_size) + paddingTop + paddingBottom)/int(pool_stride) + 1)
				result_col = int((input_pool_shape[3] - int(pool_size) + paddingLeft + paddingRight)/int(pool_stride) + 1)
				output_pool_shape = (input_pool_shape[0], input_pool_shape[1], result_row, result_col)
				output_pool_info = O.helper.make_tensor_value_info('output_pool_info'+testType, O.TensorProto.FLOAT, list(output_pool_shape))
				
				pool_node = O.helper.make_node(
				op_type='MaxPool',
				inputs=[input_str_info],
				outputs=['output_pool_info'+testType],
				name=testType,
				kernel_shape=[int(pool_size), int(pool_size)],
				pads=padding_info,
				strides=[int(pool_stride), int(pool_stride)]
				)
		elif op_mode == '1':
			# min pool
			pass
		elif op_mode == '2':
			# average pool
			result_row = int((input_pool_shape[2] - int(pool_size) + paddingTop + paddingBottom)/int(pool_stride) + 1)
			result_col = int((input_pool_shape[3] - int(pool_size) + paddingLeft + paddingRight)/int(pool_stride) + 1)
			output_pool_shape = (input_pool_shape[0], input_pool_shape[1], result_row, result_col)
			output_pool_info = O.helper.make_tensor_value_info('output_pool_info'+testType, O.TensorProto.FLOAT, list(output_pool_shape))

			pool_node = O.helper.make_node(
			op_type='AveragePool',
			inputs=[input_str_info],
			outputs=['output_pool_info'+testType],
			name=testType,
			kernel_shape=[int(pool_size), int(pool_size)],
			pads=padding_info,
			strides=[int(pool_stride), int(pool_stride)]
			)
		node_lst.append(pool_node)
		value_info_lst.append(output_pool_info)
		results = [node_lst, value_info_lst, output_pool_shape,'output_pool_info'+testType]
		return results
	return poolingLayer

def getModel(layer):
	#########################
    #   construct graph     #
    #########################
    graph_def = O.helper.make_graph(
        layer.node_list,
        layer.cfgDict["test_case_notes"] + '_onnx',
        layer.values_in,
        layer.values_out,
        value_info=layer.values_info,
        )
    #########################
    #      build model      #
    #########################
    onnx_model = O.helper.make_model(graph_def, producer_name='Kneron')
    return onnx_model