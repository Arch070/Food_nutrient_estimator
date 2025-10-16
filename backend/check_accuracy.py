# check_accuracy.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.nn as nn

data_dir = "food_images"  
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

full_dataset = datasets.ImageFolder(data_dir, transform=val_transforms)

val_size = int(0.2 * len(full_dataset))  
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
food_model_path = "food_model.pth"
checkpoint = torch.load(food_model_path, map_location=device)

food_model = models.resnet18(weights=None)
num_features = food_model.fc.in_features
food_model.fc = nn.Linear(num_features, len(checkpoint['classes']))
food_model.load_state_dict(checkpoint['model_state_dict'])
food_model.eval().to(device)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = food_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")

