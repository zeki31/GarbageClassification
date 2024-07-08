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
categories_primitive = cat_dict.iloc[:,3].unique()

# ['燃やすごみ','燃えないごみ','スプレー缶','乾電池','プラスチック製容器包装','缶,びん,ペットボトル', '小さな金属類', '古紙','古布','粗大ごみ','小型家電','市では取り扱えないもの']... 
# 30 categories


# Base directory where you want to create the folder structure
base_directory = os.path.join(work_dir, 'img_single')

'''
# Function to create directories
def create_directories(base_path, structure):
    for folder in structure:
        # Create main folder
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)

folder_structure = categories

# Create the directories
create_directories(base_directory, folder_structure)

print("Folder structure created successfully!")
'''


def list_folders(directory):
    """
    List all folders in a given directory.

    Parameters:
    directory (str): Path to the directory to list folders from.

    Returns:
    list: A list of folder names in the given directory.
    """
    # Ensure the provided path is absolute
    directory = os.path.abspath(directory)
    
    # Get all items in the directory
    all_items = os.listdir(directory)
    
    # Filter out non-directories
    folders = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]
    
    return folders

folder_list = list_folders(base_directory)

categories = folder_list


paths = {cat: os.listdir(os.path.join(base_directory, cat)) for cat in categories}
# Function to load and preprocess images
def load_images(path, category):
    images = []
    labels = []
    for img_name in path:
        img_path = os.path.join(base_directory, category, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(category)
    return images, labels

# Load all images
all_images = []
all_labels = []

for i, category in enumerate(categories):
    images, labels = load_images(paths[category], category)
    all_images.extend(images)
    all_labels.extend([i] * len(images))

# Convert to numpy arrays
X = np.array(all_images)
y = np.array(all_labels)

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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define the model
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(GarbageClassifier, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        
        # Freeze the parameters of the feature layers
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        
        # Remove the original classifier
        self.vgg16.classifier = nn.Identity()
        
        # Add a new classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256),  # 512 is the number of output channels from VGG16's last conv layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Add adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.vgg16.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GarbageClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# Training loop
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Print epoch statistics
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

# Function to classify a single image
def separately(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)
    
    return categories[predicted.item()]



# Save the model
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    # Add any other information you want to save
}, os.path.join(work_dir, 'garbage_classifier.pth'))

'''
# Load the checkpoint
checkpoint = torch.load('garbage_classifier_checkpoint.pth')

# Initialize the model architecture
model = GarbageClassifier()

# Load the model state dictionary
model.load_state_dict(checkpoint['model_state_dict'])

# load the optimizer state
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()  # Set the model to evaluation mode
'''


# Test the classifier on a single image
test_img_path = os.path.join(base_directory, categories[1], paths[categories[1]][10])
img = cv2.imread(test_img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
print(separately(img))
