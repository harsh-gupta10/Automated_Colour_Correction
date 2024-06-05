import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
import cv2
import os
from PIL import Image
from featureextraction import calculate_exposure_with_cv2, calculate_hue_with_cv2, calculate_saturation_with_cv2

# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, raw_dir, enhanced_dir, transform=None):
        self.raw_dir = raw_dir
        self.enhanced_dir = enhanced_dir
        self.transform = transform
        self.images = os.listdir(raw_dir)[:800]

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
        
        enhanced_values = torch.tensor(extract_values(enhanced_image), dtype=torch.float)  
        
        return raw_image, enhanced_values

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define the dataset and dataloader
dataset = ImageDataset('./raw', './c', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model
model = resnet18(pretrained=True)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 3),
    torch.nn.Sigmoid()  # Apply Sigmoid to ensure outputs are in [0, 1]
)

# Move model and data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training loop remains the same

def extract_values(image):
    exposure = calculate_exposure_with_cv2(image)
    saturation = calculate_saturation_with_cv2(image)
    hue = calculate_hue_with_cv2(image)
    return [hue, saturation, exposure]


# Proceed with the training as before



def train(model, dataloader, criterion, optimizer, num_epochs=10):
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
torch.save(model.state_dict(), 'trained_model.pth')

