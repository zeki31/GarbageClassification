import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd

# Set the working directory
work_dir = '~/Downloads/proj_garb_classify/'
work_dir = os.path.expanduser(work_dir)  # Expand the ~ to the full home directory path


# Define paths
cat_dict = pd.read_csv(os.path.join(work_dir, 'cat_garbage_yoko.csv'))
categories = cat_dict.iloc[:,3].unique()

# Base directory where you want to create the folder structure
base_directory = os.path.join(work_dir, 'img')

'''
# Function to create directories
def create_directories(base_path, structure):
    for folder, subfolders in structure.items():
        # Create main folder
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        
        # Create subfolders
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)

# Assuming your CSV has columns '出し方' and '品目'
folder_structure = cat_dict.groupby('出し方')['品目'].apply(list).to_dict()

# Create the directories
create_directories(base_directory, folder_structure)

print("Folder structure created successfully!")
'''

# Function to load and preprocess images
def load_images(base_dir):
    images = []
    labels = []
    label_map = {}
    
    for label_id, category in enumerate(os.listdir(base_dir)):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            label_map[label_id] = category
            for subfolder in os.listdir(category_path):
                subfolder_path = os.path.join(category_path, subfolder)
                if os.path.isdir(subfolder_path):
                    for img_name in os.listdir(subfolder_path):
                        img_path = os.path.join(subfolder_path, img_name)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (224, 224))
                            images.append(img)
                            labels.append(label_id)
    
    return np.array(images), np.array(labels), label_map

# Load all images
X, y, label_map = load_images(base_directory)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a custom dataset
class GarbageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and dataloaders
train_dataset = GarbageDataset(X_train, y_train, transform=transform)
test_dataset = GarbageDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=len(label_map)):
        super(GarbageClassifier, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        self.vgg16.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.vgg16.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GarbageClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Save the model
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, os.path.join(work_dir, 'garbage_classifier.pth'))

# Function to classify a single image
def classify_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)
    
    return list(label_map.values())[predicted.item()]

# Test the classifier on a single image
test_img_path = os.path.join(base_directory, '小さな金属類', '鉄板', 'download.jpg')  # Adjust this path as needed
img = cv2.imread(test_img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
print(f"Predicted category: {classify_image(test_img_path)}")
