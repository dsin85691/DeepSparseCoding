import onnx
import tensorflow as tf
import torch
import argparse
from onnx_tf.backend import prepare
from torch.autograd import Variable
import DeepSparseCoding.utils.loaders as loaders
from DeepSparseCoding.models.ensemble_model import NetTensorFlowWrapper

input1 = (3,32,32)

parser = argparse.ArgumentParser()
parser.add_argument('--param_file', help='Path to the parameter file')

args = parser.parse_args()
param_file = args.param_file

params = loaders.load_params(param_file)
setattr(params, "epoch_size", 60000)
setattr(params, "num_val_images", 0)
setattr(params, "num_test_images", 10000)
setattr(params, "data_shape", [28, 28, 1])
EnsembleModel1 = loaders.load_model(params.model_type, params)
EnsembleModel1.load_state_dict(torch.load(r"C:\Users\12679\CIFAR10_demo\CIFAR10_demo\CIFAR10\lca_768_mlp_mnist_latest_checkpoint_v0.pt"))
transposed_EnsembleModel1 = NetTensorFlowWrapper(EnsembleModel1)
new_dict = torch.load(r"C:\Users\12679\CIFAR10_demo\CIFAR10_demo\CIFAR10\lca_768_mlp_mnist_latest_checkpoint_v0.pt")
print(new_dict)

# net_wrapper_model.load_state_dict(torch.load(r"C:\Users\12679\CIFAR10_demo\CIFAR10_demo\CIFAR10\cifar_net.pt"))
# print(net_wrapper_model)
batch_size = 1
x = Variable(torch.randn((1, 28, 28, 1), requires_grad=True))
torch.onnx.export(transposed_EnsembleModel1, x, f="onnx_model.onnx", verbose=False, output_names=['output'])
#
onnx_model = onnx.load("onnx_model.onnx")
onnx.checker.check_model(onnx_model)
tf_rep = prepare(onnx_model)
#
#
print('inputs:', tf_rep.inputs)
print('outputs:', tf_rep.outputs)
#
print('tensor_dict:')
print(tf_rep.tensor_dict)
print(dir(tf_rep))
#
tf_rep.export_graph(r"C:\Users\12679\CIFAR10_demo\CIFAR10_demo\CIFAR10\output\new_saved_model.pb")
#
converter = tf.lite.TFLiteConverter.from_saved_model(r"C:\Users\12679\CIFAR10_demo\CIFAR10_demo\CIFAR10\output\new_saved_model.pb")

tflite_model = converter.convert()
#
with open(r"C:\Users\12679\DeepSparseCoding\DeepSparseCoding\transposed_model.tflite", "wb") as f:
     f.write(tflite_model)
