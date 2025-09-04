import os
import cv2 
from torchvision import models, transforms
import torch 
import numpy as np

from ml_functions.models import SimpleCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelContainer:
    def __init__(self, model_path, IMG_SIZE=(224, 224)):
        self.model = self.get_model(model_path)
        self.IMG_SIZE = IMG_SIZE
        # The transformation is now a simple composition to be applied per-image
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_model(self, model_path):
        # Ensure the checkpoints folder and model file exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

        # Instantiate the model with the correct input size
        IMG_SIZE = (224, 224)
        model = SimpleCNN(input_size=IMG_SIZE).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        return model 
    
    def decode_image(self, contents):
        image_np = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        # Ensure image is valid before converting color
        if image is None:
            raise ValueError("Could not decode image from contents.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _transform_batch(self, images):
        """Transforms a list of NumPy images into a single PyTorch tensor."""
        transformed_images = [self.transform(img) for img in images]
        # Stack the list of tensors into a single batch tensor
        batch_tensor = torch.stack(transformed_images).to(device)
        return batch_tensor

    def run_inference(self, images):
        """
        Runs inference on a batch of images.

        Args:
            images (list): A list of NumPy image arrays.

        Returns:
            list: A list of probability scores for each image in the batch.
        """
        if not isinstance(images, list) or not images:
            raise ValueError("Input must be a non-empty list of images.")
            
        # Transform the batch of images
        batch_tensor = self._transform_batch(images)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            # Apply sigmoid to get probabilities for the entire batch
            probabilities = torch.sigmoid(outputs).squeeze().tolist()

        # Handle the case of a single image gracefully
        if len(images) == 1:
            probabilities = [probabilities]

        return probabilities