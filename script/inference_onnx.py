import onnxruntime as rt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets

import matplotlib.pyplot as plt

from PIL import Image

sess = rt.InferenceSession("model/model_3layer.onnx")
input_name = sess.get_inputs()[0].name
print(input_name)

image = Image.open('./data/test/2.png')
image_data = np.array(image)
# image_data.reshape(1,1,28,28)
image_data = image_data.astype(np.float32)/255.0

print(image_data.shape)
# image_data = np.expand_dims(image_data,axis=0)
# print(image_data.shape)

image_data = image_data[None,None,:,:]
print(image_data.shape)

outputs = sess.run(None,{input_name:image_data})
outputs = np.array(outputs)
print(outputs.shape)
print(outputs)
# bb = outputs.argmax()
# print(bb)
# max_index = np.unravel_index(outputs.argmax(), outputs.shape)
_,r,c = np.where(outputs == np.max(outputs))
label = 1
prediction = c
print(f"Label: {label}, Predicted: {prediction}")
plt.imshow(image, cmap='gray')
plt.title(f"Label: {label}, Predicted: {prediction}")
plt.show()

# # Download test data from open datasets.
# test_data = datasets.MNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )

# # Create data loader.
# test_dataloader = DataLoader(test_data, batch_size=1)

# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_dataloader:
#         images = images.numpy()
#         labels = labels.numpy()
#         outputs = sess.run(None,{input_name:images})[0]
#         outputs = np.array(outputs)
#         outputs.reshape(-1)
#         print(outputs)
#         predicted = np.where(outputs == np.max(outputs))
#         print(predicted[1])

#         # Visualize the image and its predicted result
#         for i in range(len(images)):
#             image = images[i] #.cpu()
#             label = labels[i] #.cpu()
#             prediction = predicted[1]

#             plt.imshow(image.squeeze(), cmap='gray')
#             plt.title(f"Label: {label}, Predicted: {prediction}")
#             plt.show()