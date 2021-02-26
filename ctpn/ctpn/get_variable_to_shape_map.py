import os
from tensorflow.python import pywrap_tensorflow

current_path = os.getcwd()
model_dir = os.path.join(current_path, '/work/pycharm_python/reconstruction_ctpn/')
# model_dir = os.path.join(current_path, '/models')
checkpoint_path = os.path.join(model_dir, 'ctpn_5.ckpt')
# checkpoint_path = os.path.join(model_dir, 'vgg_16.ckpt')
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key)) # 打印变量的值，对我们查找问题没啥影响，打印出来反而影响找问题

