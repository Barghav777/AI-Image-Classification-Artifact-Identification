import argparse
import joblib
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Load your model
model = joblib.load("EfficientNet_model.pkl")

def predict_image(image_path):
    # Preprocess the input image
    img = load_img(image_path, target_size=(32, 32))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Perform prediction
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    
    # Determine class
    predicted_class = 'real' if probability >= 0.5 else 'fake'
    return predicted_class, probability

# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict whether an image is real or fake.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    
    args = parser.parse_args()
    image_path = args.image_path
    
    try:
        predicted_class, probability = predict_image(image_path)
        print(f"The image is predicted to be: {predicted_class}")
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
