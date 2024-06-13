import torch
import torch.utils
import os
from ZKNNNet import ZKNNNet_3Layer,ZKNNNet_5Layer,ZKNNNet_Conv

device = "cpu"
print("Using {} device".format(device))
# model_3Layer = ZKNNNet_3Layer()
# if os.path.exists('./model/model_3layer.pth'):
#     model_3Layer.load_state_dict(torch.load('./model/model_3layer.pth'))
# model_3Layer = model_3Layer.to(device)

# model_3Layer.eval()

# # export pytorch model to onnx
# torch.onnx.export(model_3Layer, torch.randn(1, 1, 28, 28), './model/model_3layer.onnx', verbose=True)

model_5Layer = ZKNNNet_5Layer()
if os.path.exists('./model/model_5layer.pth'):
    model_5Layer.load_state_dict(torch.load('./model/model_5layer.pth'))
model_5Layer = model_5Layer.to(device)
model_5Layer.eval()
torch.onnx.export(model_5Layer,torch.randn(1,1,28,28),'./model/model_5layer.onnx',verbose=True)

model_conv = ZKNNNet_Conv()
if os.path.exists('./model/model_conv.pth'):
    model_conv.load_state_dict(torch.load('./model/model_conv.pth'))
model_conv = model_conv.to(device)
model_conv.eval()
torch.onnx.export(model_conv,torch.randn(1,1,28,28),'./model/model_conv.onnx',verbose=True)