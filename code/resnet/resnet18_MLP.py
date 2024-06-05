import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
import cv2
import os
from PIL import Image
from featureextraction import extract_hsv

# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, raw_dir, enhanced_dir, transform=None):
        self.raw_dir = raw_dir
        self.enhanced_dir = enhanced_dir
        self.transform = transform
        self.images = os.listdir(raw_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        raw_image_path = os.path.join(self.raw_dir, self.images[idx])
        enhanced_image_path = os.path.join(self.enhanced_dir, self.images[idx])
        
        # Load images with OpenCV
        raw_image = cv2.imread(raw_image_path)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        enhanced_image = cv2.imread(enhanced_image_path)
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

        # Apply transformations (if any)
        if self.transform:
            raw_image = Image.fromarray(raw_image)
            raw_image = self.transform(raw_image)
        
        enhanced_values = torch.tensor(extract_hsv(enhanced_image), dtype=torch.float)  
        
        return raw_image, enhanced_values

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define the dataset and dataloader
dataset = ImageDataset('./raw', '/media/ishangupta/Crucial X6/b', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


class ResNet18WithMLP(torch.nn.Module):
    def __init__(self):
        super(ResNet18WithMLP, self).__init__()
        # Load pre-trained ResNet18 and remove its fully connected layer
        self.features = resnet18()
        number_features=self.features.fc.in_features
        self.features.fc = torch.nn.Identity()  # Remove the last FC layer
        
        # Define an MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(number_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 3),
            torch.nn.Sigmoid()  # Ensure outputs are in [0, 1]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.mlp(x)
        return x



# Model
model = ResNet18WithMLP()


# Move model and data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training loop remains the same


# Proceed with the training as before

def train(model, dataloader, criterion, optimizer, num_epochs=6):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs,targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

train(model, dataloader, criterion, optimizer)
# After the training loop
# Save the trained model
torch.save(model.state_dict(), 'trained_model_b.pth')

