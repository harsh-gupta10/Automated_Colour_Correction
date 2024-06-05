import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from PIL import Image


# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, raw_dir, edited_dir, transform=None):
        self.raw_dir = raw_dir
        self.edited_dir = edited_dir
        self.transform = transform
        self.raw_images = sorted(os.listdir(raw_dir))
        self.edited_images = sorted(os.listdir(edited_dir))

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, idx):
        raw_path = os.path.join(self.raw_dir, self.raw_images[idx])
        edited_path = os.path.join(self.edited_dir, self.edited_images[idx])

        raw_image = Image.open(raw_path).convert("RGB")
        edited_image = Image.open(edited_path).convert("RGB")

        if self.transform:
            raw_image = self.transform(raw_image)
            edited_image = self.transform(edited_image)

        return raw_image, edited_image


# U-Net generator architecture
class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_downs):
        super(UNetGenerator, self).__init__()

        # Encoder (downsampling)
        self.enc_conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.enc_conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)

        # Decoder (upsampling)
        self.dec_conv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv2 = nn.ConvTranspose2d(
            1024, 256, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv3 = nn.ConvTranspose2d(
            512, 128, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv5 = nn.ConvTranspose2d(
            128, out_channels, kernel_size=4, stride=2, padding=1
        )

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        enc1 = self.relu(self.enc_conv1(x))
        enc2 = self.relu(self.enc_conv2(enc1))
        enc3 = self.relu(self.enc_conv3(enc2))
        enc4 = self.relu(self.enc_conv4(enc3))
        enc5 = self.relu(self.enc_conv5(enc4))

        # Decoder
        dec1 = self.relu(self.dec_conv1(enc5))
        dec1 = torch.cat([dec1, enc4], dim=1)
        dec2 = self.relu(self.dec_conv2(dec1))
        dec2 = torch.cat([dec2, enc3], dim=1)
        dec3 = self.relu(self.dec_conv3(dec2))
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec4 = self.relu(self.dec_conv4(dec3))
        dec4 = torch.cat([dec4, enc1], dim=1)
        dec5 = self.tanh(self.dec_conv5(dec4))

        return dec5


# PatchGAN discriminator architecture
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels, num_filters=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()

        layers = [
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        for i in range(1, n_layers):
            layers.append(
                nn.Conv2d(
                    num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1
                )
            )
            layers.append(nn.BatchNorm2d(num_filters * 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            num_filters *= 2

        layers.append(nn.Conv2d(num_filters, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Training parameters
batch_size = 4
num_epochs = 4
learning_rate = 0.0002
lambda_l1 = 100

# Data directories
raw_dir = "data/train/raw"
edited_dir = "data/train/c"

# Image transformations
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Create dataset and dataloader
dataset = ImageDataset(raw_dir, edited_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize generator and discriminator
generator = UNetGenerator(3, 3, num_downs=8)
discriminator = PatchGANDiscriminator(6)

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# Define loss functions and optimizers
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(
    discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
)

# Training loop


# # Test the trained model
# generator.eval()
# test_image_path = "data/val/raw_o/a0026-kme_391.jpg"
# test_image = cv2.imread(test_image_path)
# test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
# test_image = Image.fromarray(test_image)
# test_image = transform(test_image).unsqueeze(0).to(device)

# with torch.no_grad():
#     output_image = generator(test_image)
#     output_image = output_image.squeeze().cpu().numpy()
#     output_image = np.transpose(output_image, (1, 2, 0))
#     output_image = (output_image + 1) / 2
#     output_image = (output_image * 255).astype(np.uint8)
#     output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
#     # cv2.imshow("Output Image", output_image)
#     cv2.imwrite("output_image.jpg", output_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()


def test_model(model_path, test_folder, output_folder, device):
    # Load the trained generator model
    generator = UNetGenerator(3, 3, num_downs=8)
    generator.load_state_dict(torch.load(model_path))
    generator.to(device)
    generator.eval()

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of image files in the test folder
    image_files = [
        f
        for f in os.listdir(test_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Process each image file
    for image_file in image_files:
        # Load and preprocess the image
        image_path = os.path.join(test_folder, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0).to(device)

        # Generate the output image
        with torch.no_grad():
            output_image = generator(image)
            output_image = output_image.squeeze().cpu().numpy()
            output_image = np.transpose(output_image, (1, 2, 0))
            output_image = (output_image + 1) / 2
            output_image = (output_image * 255).astype(np.uint8)
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        # Save the output image
        output_path = os.path.join(output_folder, f"output_{image_file}")
        cv2.imwrite(output_path, output_image)

    print(f"Processed {len(image_files)} images. Output saved in {output_folder}.")


# ... (previous code remains the same) ...

# Specify the paths for testing
model_path = "generator_epoch_100.pth"  # Path to the trained generator model
test_folder = "data/val/raw_o"  # Path to the folder containing test images
output_folder = (
    "output"  # Path to the folder where output images will be saved
)

# Test the trained model on the test folder
test_model(model_path, test_folder, output_folder, device)
