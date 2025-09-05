import os
import cv2 
from torchvision import models, transforms
import torch 
import numpy as np

from ml_functions.models import SimpleCNN, get_pretrainded_mobilenetv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelContainer:
    def __init__(self, model_type, model_path, IMG_SIZE=(224, 224)):
        self.IMG_SIZE = IMG_SIZE

        self.model = self.get_model(model_type, model_path)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_model(self, model_type, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

        if model_type == 'simple_cnn':
            model = SimpleCNN(input_size=self.IMG_SIZE).to(device)
        elif model_type == 'mobilenet_v2':
            model = get_pretrainded_mobilenetv2(device, pretrained=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported types are 'simple_cnn' and 'mobilenet_v2'.")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        return model 
    
    def decode_image(self, contents):
        image_np = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from contents.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _transform_batch(self, images):
        transformed_images = [self.transform(img) for img in images]
        batch_tensor = torch.stack(transformed_images).to(device)
        return batch_tensor

    def run_inference(self, images):
        if not isinstance(images, list) or not images:
            raise ValueError("Input must be a non-empty list of images.")
            
        batch_tensor = self._transform_batch(images)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.sigmoid(outputs).squeeze().tolist()

        # Handle the case of a single image gracefully
        if len(images) == 1:
            probabilities = [probabilities]

        return probabilities