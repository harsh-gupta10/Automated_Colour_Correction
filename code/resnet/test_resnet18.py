import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
import cv2
import os
from PIL import Image

from featureapplication import adjust_image
# Assuming calculate_exposure_with_cv2, calculate_hue_with_cv2, calculate_saturation_with_cv2 are defined elsewhere
from featureextraction import calculate_exposure_with_cv2, calculate_hue_with_cv2, calculate_saturation_with_cv2

def extract_values(image):
    exposure = calculate_exposure_with_cv2(image)
    saturation = calculate_saturation_with_cv2(image)
    hue = calculate_hue_with_cv2(image)
    return [hue, saturation, exposure]


# Custom Dataset for Testing
class TestImageDataset(Dataset):
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
        
        raw_image = cv2.imread(raw_image_path)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        enhanced_image = cv2.imread(enhanced_image_path)
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            raw_image = Image.fromarray(raw_image)
            raw_image = self.transform(raw_image)
          
        
        enhanced_values = torch.tensor(extract_values(enhanced_image), dtype=torch.float)
        
        return raw_image, enhanced_values

# Transformations, same as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the trained model
model = resnet18(pretrained=True)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 3),
    torch.nn.Sigmoid()
)

# Load model weights
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()  # Set the model to evaluation mode

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the test dataset and dataloader
test_dataset = TestImageDataset('./raw_test', './c_test', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# MSE Evaluation
criterion = torch.nn.MSELoss()
total_loss = 0.0

# Predictions list
predictions=[]
with torch.no_grad():  # No need to calculate gradients during evaluation
    for inputs, targets in test_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())
        loss = criterion(outputs, targets)
        total_loss += loss.item()

average_loss = total_loss / len(test_dataloader)
print(f'Average MSE Loss on Test Set: {average_loss}')


images = os.listdir('./raw_test')



for idx in range(0,200):
    raw_image_path = os.path.join('./raw_test', images[idx])
    #enhanced_image_path = os.path.join('./c_test',images[idx])
    raw_image=cv2.imread(raw_image_path)
    #enhanced_image=cv2.imread(enhanced_image_path)
    predicted_image=adjust_image(raw_image,predictions[idx][0],predictions[idx][1],predictions[idx][2]) 
    predicted_image_path=os.path.join('./predicted_images',images[idx])
    cv2.imwrite(predicted_image_path,predicted_image) 

# Assuming the extract_values function is defined the same way as in the training script

