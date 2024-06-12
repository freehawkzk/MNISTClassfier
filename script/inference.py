import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
from ZKNNNet import ZKNNNet_3Layer

import matplotlib.pyplot as plt

# Get cpu or gpu device for inference.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device for inference".format(device))

# Load the trained model
model = ZKNNNet_3Layer()
model.load_state_dict(torch.load("model/model_3layer.pth"))
model.to(device)
model.eval()

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Create data loader.
test_dataloader = DataLoader(test_data, batch_size=64)



# Perform inference
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Visualize the image and its predicted result
        for i in range(len(images)):
            image = images[i].cpu()
            label = labels[i].cpu()
            prediction = predicted[i].cpu()

            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f"Label: {label}, Predicted: {prediction}")
            plt.show()

    accuracy = 100 * correct / total
    print("Accuracy on test set: {:.2f}%".format(accuracy))