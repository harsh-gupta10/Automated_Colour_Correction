import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
import cv2
import os
from PIL import Image

from featureapplication import adjust_image
from featureextraction import extract_hsv

RAW_DIR='./test_images'

def extract_values(image):
    return extract_hsv(image)


# Custom Dataset for Testing
class TestImageDataset(Dataset):
    def __init__(self, raw_dir,transform=None):
        self.raw_dir = raw_dir
        self.transform = transform
        self.images = os.listdir(raw_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        raw_image_path = os.path.join(self.raw_dir, self.images[idx])
        
        raw_image = cv2.imread(raw_image_path)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        

        if self.transform:
            raw_image = Image.fromarray(raw_image)
            raw_image = self.transform(raw_image)
          
        return raw_image

# Transformations, same as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the trained model
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

# Load model weights
model.load_state_dict(torch.load('trained_model_c.pth'))
model.eval()  # Set the model to evaluation mode

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the test dataset and dataloader
test_dataset = TestImageDataset(RAW_DIR,transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)



# Predictions list
predictions=[]
with torch.no_grad():  # No need to calculate gradients during evaluation
    for inputs in test_dataloader:
        inputs= inputs.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())
    



images = os.listdir(RAW_DIR)
for idx in range(len(images)):
    raw_image_path = os.path.join(RAW_DIR, images[idx])
    raw_image=cv2.imread(raw_image_path)
    predicted_image=adjust_image(raw_image,predictions[idx][0],predictions[idx][1],predictions[idx][2]) 
    predicted_image_path=os.path.join('./predicted_images_final/c',images[idx])
    cv2.imwrite(predicted_image_path,predicted_image) 

