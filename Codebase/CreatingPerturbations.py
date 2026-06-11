# Create perturbed images for adversarial perturbations training

import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

def fgsm_attack(image, epsilon, gradient):
    """
    Generates an adversarial example using FGSM.
    """
    sign_data_grad = gradient.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure values are in [0,1]
    return perturbed_image

def create_adversarial_images(root_dir, num_images, epsilon, output_dir="adversarial_images"):
    """
    Creates adversarially perturbed images using FGSM.
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load a pre-trained model
    model = resnet18(pretrained=True).to(device)
    model.eval()  # Set model to evaluation mode

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load dataset
    dataset = ImageFolder(root=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    count = 100000
    for i, (image, label) in enumerate(dataloader):
        # if count >= num_images:
        #     break

        image, label = image.to(device), label.to(device)
        image.requires_grad = True

        # Forward pass
        output = model(image)
        loss = torch.nn.functional.cross_entropy(output, label)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Get gradient and apply FGSM
        gradient = image.grad.data
        perturbed_image = fgsm_attack(image, epsilon, gradient)

        # Save the perturbed image
        perturbed_image_np = perturbed_image.squeeze().cpu().detach().numpy()
        perturbed_image_pil = transforms.ToPILImage()(perturbed_image.squeeze())
        class_name = dataset.classes[label.item()]
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
        output_path = os.path.join(output_dir, class_name, f"perturbed_{count}.png")
        perturbed_image_pil.save(output_path)

        print(f"Saved {output_path}")
        count += 1
        # break

root_dir = "path_to_your_folder_containing_images"
num_images = 0
epsilon = 0.1
output_dir = "path_to_output_directory"

create_adversarial_images(root_dir, num_images, epsilon, output_dir)