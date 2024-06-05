import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.utils import save_image
from PIL import Image
import os

BATCH_SIZE = 4
IMAGE_SIZE = 256
EPOCHS = 150
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)
LAMBDA = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "images"


os.makedirs(OUTPUT_DIR, exist_ok=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv7 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv8 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        x6 = F.relu(self.conv6(x5))
        x7 = F.relu(self.conv7(x6))
        x8 = F.relu(self.conv8(x7))

        x9 = F.relu(self.deconv1(x8))
        x9 = torch.cat([x9, x7], dim=1)
        x10 = F.relu(self.deconv2(x9))
        x10 = torch.cat([x10, x6], dim=1)
        x11 = F.relu(self.deconv3(x10))
        x11 = torch.cat([x11, x5], dim=1)
        x12 = F.relu(self.deconv4(x11))
        x12 = torch.cat([x12, x4], dim=1)
        x13 = F.relu(self.deconv5(x12))
        x13 = torch.cat([x13, x3], dim=1)
        x14 = F.relu(self.deconv6(x13))
        x14 = torch.cat([x14, x2], dim=1)
        x15 = F.relu(self.deconv7(x14))
        x15 = torch.cat([x15, x1], dim=1)
        x16 = F.tanh(self.deconv8(x15))

        return x16

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2)
        x5 = F.sigmoid(self.conv5(x4))
        return x5

criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()


generator = Generator().to(device)
discriminator = Discriminator().to(device)


optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir_raw, root_dir_filtered, transform=None):
        self.root_dir_raw = root_dir_raw
        self.root_dir_filtered = root_dir_filtered
        self.transform = transform

        self.raw_images = os.listdir(root_dir_raw)
        self.filtered_images = os.listdir(root_dir_filtered)

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, idx):
        raw_img_name = os.path.join(self.root_dir_raw, self.raw_images[idx])
        filtered_img_name = os.path.join(self.root_dir_filtered, self.filtered_images[idx])
        
        raw_image = Image.open(raw_img_name).convert("RGB")
        filtered_image = Image.open(filtered_img_name).convert("RGB")

        if self.transform:
            raw_image = self.transform(raw_image)
            filtered_image = self.transform(filtered_image)

        return raw_image, filtered_image


train_dataset = CustomDataset(root_dir_raw='/home/acer/raw',
                              root_dir_filtered='/home/acer/a',
                              transform=transform)

test_dataset = CustomDataset(root_dir_raw='/home/acer/raw',
                             root_dir_filtered='/home/acer/a',
                             transform=transform)


train_indices = list(range(200))
test_indices = list(range(200,300))


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_indices))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(test_indices))


for epoch in range(EPOCHS):
    for batch_idx, (real_images, target_images) in enumerate(train_loader):
        
        real_images = real_images.to(device)
        target_images = target_images.to(device)

       
        optimizer_G.zero_grad()

        fake_images = generator(real_images)

        pred_fake = discriminator(fake_images)
        loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

        loss_L1 = criterion_L1(fake_images, target_images)

        loss_G = loss_GAN + LAMBDA * loss_L1

        loss_G.backward()
        optimizer_G.step()

       
        optimizer_D.zero_grad()

        pred_real = discriminator(target_images)
        loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

        pred_fake = discriminator(fake_images.detach())
        loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()
        

        if batch_idx % 100 == 0:
            print(
                f"[Epoch {epoch}/{EPOCHS}] [Batch {batch_idx}/{len(train_loader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]"
            )

generator.eval() 
test_loss_G = 0
test_loss_L1 = 0

os.makedirs("test_images", exist_ok=True)

with torch.no_grad():
    for batch_idx, (real_images, target_images) in enumerate(test_loader):
       
        real_images = real_images.to(device)
        target_images = target_images.to(device)

        
        fake_images = generator(real_images)

        
        pred_fake = discriminator(fake_images)
        test_loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

        test_loss_L1 += criterion_L1(fake_images, target_images)

        test_loss_G += test_loss_GAN.item()

        
        for idx, (image, target) in enumerate(zip(fake_images, target_images)):
            
            filename = test_dataset.filtered_images[batch_idx * BATCH_SIZE + idx]

            
            save_image(image, f"test_images/{filename}", normalize=True)

    test_loss_G /= len(test_loader)
    test_loss_L1 /= len(test_loader)

    print(f"Test Loss GAN: {test_loss_G}, Test Loss L1: {test_loss_L1}")

