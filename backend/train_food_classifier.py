# train_food_classifier.py
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

data_dir = "food_images"           
save_model_path = "food_model.pth" 

batch_size = 16
epochs = 5
lr = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2. Data transforms & loader
# -------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# 3. Model setup
# -------------------------------
model = models.resnet18(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))  # Output = number of food classes
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=lr)

# -------------------------------
# 4. Training loop
# -------------------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# -------------------------------
# 5. Save model
# -------------------------------
torch.save({
    'model_state_dict': model.state_dict(),
    'classes': train_dataset.classes
}, save_model_path)

print(f"Training complete! Model saved to {save_model_path}")
