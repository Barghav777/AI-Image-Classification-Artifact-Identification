import sys
import os
import joblib
import torch
import shap
import cv2
import json
import numpy as np
import xgboost as xgb
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import requests

import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

from transformers import ViTFeatureExtractor, ViTModel, AutoProcessor, LlavaForConditionalGeneration
from torchvision import transforms
from huggingface_hub import login

# Updated path to include Codebase
efficientnet_model = joblib.load("Codebase/Task_1/EfficientNet_model.pkl")

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")

llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

def image_classification(image_path):
    def predict_image(image_path):
        img = load_img(image_path, target_size=(32, 32))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        prediction = efficientnet_model.predict(img_array)
        probability = prediction[0][0]
        
        # Determine the class based on the probability
        if probability >= 0.5:
            predicted_class = 'real'
        else:
            predicted_class = 'fake'
        
        return predicted_class

    predicted_class = predict_image(image_path)
    return predicted_class

def artifact_detection(image_path, hf_token):
    login(token=hf_token) # Replace 'your_huggingface_token' with your actual token

    # Updated paths to include Codebase
    models_directory = "Codebase/Task_2/models/xgb_model_params/xgb_models/"
    models_directory1 = "Codebase/Task_2/models/xgb_models-no_params/xgb_models/"
    models_directory2 = "Codebase/Task_2/models/rf_models-20241126T171044Z-001/rf_models/"

    feature_extracting_model = "google/vit-large-patch16-224-in21k"
    explanation_generator_model = "llava-hf/llava-1.5-7b-hf"

    # Define transformation required for processing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the function: loading all models and storing in a dictionary
    def get_artifact_name(model_path):
        filename = os.path.basename(model_path)
        if "xgb_model_" in filename:
            return filename.split("xgb_model_")[1].split(".pth")[0]
        elif "rf_model_" in filename:
            return filename.split("rf_model_")[1].split(".pth")[0]
        else:
            raise ValueError("Invalid model path: must contain 'xgb_model_' or 'rf_model_'")

    # Load all models from a given directory
    def load_all_models(directory):
        models = {}
        if not os.path.exists(directory):
            return models
        for model_file in os.listdir(directory):
            if model_file.endswith(".pth"):
                model_path = os.path.join(directory, model_file)
                artifact_name = get_artifact_name(model_file)
                # Use joblib.load for .pth if it's a pickle file (ensure models were saved as joblib)
                models[artifact_name] = joblib.load(model_path)
        return models

    # Loading all models
    models = load_all_models(models_directory)
    models1 = load_all_models(models_directory1)
    models2 = load_all_models(models_directory2)

    # Extract image features
    def extract_image_features(image, transform):
        image = image.convert("RGB")
        vit_model.to(device)
        vit_model.eval()
        with torch.no_grad():
            transformed_image = transform(image).unsqueeze(0).to(device)
            outputs = vit_model(transformed_image)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embedding

    # Predict with model
    def predict_with_model(model, image_features):
        return int(model.predict(image_features)[0])

    # Check artifacts in image
    def check_artifacts_on_image(models, image_features):
        artifact_results = []
        model_list = []
        for artifact_name, m in models.items():
            if predict_with_model(m, image_features):
                artifact_results.append(artifact_name)
                model_list.append(m)
        return artifact_results, model_list

    # Predict all artifacts from all models
    def predict_artifacts(image_path):
        image = Image.open(image_path)
        image_features = extract_image_features(image, transform)
        artifact_names, models_name = check_artifacts_on_image(models, image_features)
        artifact_names1, models_name1 = check_artifacts_on_image(models1, image_features)
        artifact_names2, models_name2 = check_artifacts_on_image(models2, image_features)
        final_artifact_names = list(set(artifact_names + artifact_names1 + artifact_names2))
        return final_artifact_names, artifact_names, artifact_names1, artifact_names2, models_name, models_name1, models_name2

    # Return final artifacts detected
    def result(image):
        image_features = extract_image_features(image, transform)
        artifact_names, models_name = check_artifacts_on_image(models, image_features)
        artifact_names1, models_name1 = check_artifacts_on_image(models1, image_features)
        artifact_names2, models_name2 = check_artifacts_on_image(models2, image_features)
        final_artifact_names = list(set(artifact_names + artifact_names1 + artifact_names2))
        return final_artifact_names

    # Define the function: For returning heatmap and bounding-box added image
    # Artifacts map function with SHAP
    def artifacts_map(model, reshaped_embeddings_np, original_image, artifact_name):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(reshaped_embeddings_np)
        shap_values = np.array(shap_values)

        if isinstance(shap_values, list):
            shap_values = np.mean(shap_values, axis=0)
        elif shap_values.ndim == 3:
            shap_values = shap_values.mean(axis=-1)

        if shap_values.ndim > 1:
            shap_values = shap_values.mean(axis=1)

        num_patches = reshaped_embeddings_np.shape[0]
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            grid_h = int(np.floor(np.sqrt(num_patches)))
            grid_w = int(np.ceil(num_patches / grid_h))
        else:
            grid_h = grid_w = grid_size

        assert len(shap_values) == num_patches, (
            f"SHAP values size ({len(shap_values)}) does not match patch count ({num_patches})."
        )

        shap_grid = shap_values.reshape((grid_h, grid_w))

        if isinstance(original_image, np.ndarray):
            image_height, image_width = original_image.shape[:2]
        else:
            image_width, image_height = original_image.size
        heatmap = resize(shap_grid, (image_height, image_width))

        normalized_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

        threshold = 0.7  # Adjust this value as needed
        binary_mask = normalized_heatmap > threshold
        binary_mask = (binary_mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if isinstance(original_image, np.ndarray):
            original_image_cv = original_image.copy()
        else:
            original_image_cv = np.array(original_image)[:, :, ::-1].copy()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(original_image_cv, (x, y), (x + w, y + h), (255, 0, 0), 1)

        image_with_bb = Image.fromarray(cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB))

        return image_with_bb, normalized_heatmap, contours

    # Show artifacts: returns bounding box annotated image and heatmap
    def show_artifacts(image, artifacts, artifact_names, artifact_names1, artifact_names2, models, models1, models2):
        image1 = image
        image = image.convert("RGB")
        transformed_image = transform(image).unsqueeze(0).to(device)
        outputs = vit_model(transformed_image)

        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        attention_maps = outputs.attentions

        batch_size, num_patches, embedding_dim = patch_embeddings.shape
        reshaped_embeddings = patch_embeddings.reshape(batch_size * num_patches, embedding_dim)
        reshaped_embeddings_np = reshaped_embeddings.detach().cpu().numpy()

        if artifacts in artifact_names:
            index = artifact_names.index(artifacts)
            model_obj = models[index]
            return artifacts_map(model_obj, reshaped_embeddings_np, image1, artifacts)
        elif artifacts in artifact_names1:
            index = artifact_names1.index(artifacts)
            model_obj = models1[index]
            return artifacts_map(model_obj, reshaped_embeddings_np, image1, artifacts)
        elif artifacts in artifact_names2:
            index = artifact_names2.index(artifacts)
            model_obj = models2[index]
            return artifacts_map(model_obj, reshaped_embeddings_np, image1, artifacts)
        else:
            print("Artifact not found")
            return None, None, None

    # Show a specific artifact map
    def show(image_path, artifact):
        final_artifact_names, artifact_names, artifact_names1, artifact_names2, models_name, models_name1, models_name2 = predict_artifacts(image_path)
        image = Image.open(image_path)
        return show_artifacts(image, artifact, artifact_names, artifact_names1, artifact_names2, models_name, models_name1, models_name2)

    # Generate explanations for artifacts using LLaVA
    def artifact_with_explanation(image_path, artifact):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        llava_model.to(device)
        llava_model.eval()
        image = Image.open(image_path).convert("RGB")

        prompt = f"""USER: <image>
                Artifact name "{artifact}".
                Explain how it is present in the image.
                Your explanation must not exceed 50 words.
                ASSISTANT:
                """

        # Prepare inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate
        with torch.no_grad():
            generate_ids = llava_model.generate(**inputs, max_new_tokens=1000)

        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output[output.index("ASSISTANT:")+len("ASSISTANT:")+1:]

    # Additional helper functions for determining location
    def get_largest_bounding_box(contours):
        largest_area = 0
        largest_box = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_box = (x, y, w, h)
        return largest_box

    def get_location_description(box, image_width, image_height):
        x, y, w, h = box
        center_x = x + w/2
        center_y = y + h/2

        # Simple quadrant-based approach:
        horizontal_mid = image_width / 2
        vertical_mid = image_height / 2

        if center_x < horizontal_mid and center_y < vertical_mid:
            return "top-left"
        elif center_x >= horizontal_mid and center_y < vertical_mid:
            return "top-right"
        elif center_x < horizontal_mid and center_y >= vertical_mid:
            return "bottom-left"
        else:
            return "bottom-right"
        
    #Function for spliting the response of LLaVA model and returns list of artifact name and explanation
    def extract_details_from_response(artifact_exp):
        artifact_generated = []
        explanation_generated = []
        i = 0
        while i<len(artifact_exp):
            while i<len(artifact_exp) and artifact_exp[i:i+9]!='Artifact:':
                i+=1
            i = i+10
            substring_exp = ""
            while i<len(artifact_exp) and artifact_exp[i]!='\n':
                substring_exp+=artifact_exp[i]
            i+=1
            artifact_generated.append(substring_exp)
            while i<len(artifact_exp) and artifact_exp[i:i+12]!='Description:':
                i+=1
            i = i+13
            substring_exp = ""
            while i<len(artifact_exp) and artifact_exp[i]!='\n':
                substring_exp+=artifact_exp[i]
            i+=1
            explanation_generated.append(substring_exp)
        return artifact_generated,explanation_generated

    #Function for finding location of artifact in image
    def locating_artifact(original_image, image_with_bb, artifact_name, contours):
        image_width, image_height = original_image.size
        largest_box = get_largest_bounding_box(contours)
        if largest_box is None:
            location_desc = "in the image"
        else:
            location_desc = get_location_description(largest_box, image_width, image_height)
        return location_desc

    keys = {
        "Abruptly cut-off objects" : "Abruptly cut off objects",
        "Aliasing along high-contrast edges" : "Aliasing along high-contrast edges",
        "Anatomically_Impossible_Joint_Configurations" : "Anatomically impossible joint configurations",
        "Anatomically_Incorrect_Paw_Structures" : "Anatomically incorrect paw structures",
        "Artificial depth of field in object presentation" : "Artificial depth of field in object presentation",
        "Artificial enhancement artifacts" : "Artificial enhancement artifacts",
        "Artificial noise patterns in uniform surfaces" : "Artificial noise patterns in uniform surfaces",
        "Artificial smoothness" : "Artificial smoothness",
        "Asymmetric_features_in_naturally_symmetric_objects" : "Asymmetric features in naturally symmetric objects",
        "Biological_Asymmetry_Errors" : "Biological asymmetry errors",
        "Blurred boundaries in fine details" : "Blurred boundaries in fine details",
        "Cinematization_Effects" : "Cinematization Effects",
        "Color coherence breaks" : "Color coherence breaks",
        "Dental_Anomalies_in_Mammals" : "Dental anomalies in mammals",
        "Depth_perception_anomalies" : "Depth perception anomalies",
        "Discontinuous_surfaces" : "Discontinuous surfaces",
        "Distorted window reflections" : "Distorted window reflections",
        "Dramatic_lighting_that_defies_natural_physics" : "Dramatic lighting that defies natural physics",
        "Exaggerated_characteristic_features" : "Exaggerated characteristic features",
        "Fake depth of field" : "Fake depth of field",
        "Floating_or_disconnected_components" : "Floating or disconnected components",
        "Frequency_domain_signatures" : "Frequency domain signatures",
        "Ghosting effects_ Semi-transparent duplicates of elements" : "Ghosting effects: Semi-transparent duplicates of elements",
        "Glow_or_light_bleed_around_object_boundaries" : "Glow or light bleed around object boundaries",
        "Implausible_aerodynamic_structures" : "Implausible aerodynamic structures",
        "Impossible_Foreshortening_in_Animal_Bodies" : "Impossible foreshortening in animal bodies",
        "Impossible_mechanical_connections" : "Impossible mechanical connections",
        "Impossible_mechanical_joints" : "Impossible mechanical joints",
        "Improper_Fur_Direction_Flows" : "Improper fur direction flows",
        "Inconsistent_material_properties" : "Inconsistent material properties",
        "Inconsistent_object_boundaries" : "Inconsistent object boundaries",
        "Inconsistent_scale_of_mechanical_parts" : "Inconsistent scale of mechanical parts",
        "Inconsistent_shadow_directions" : "Inconsistent shadow directions",
        "Incorrect Skin Tones" : "Incorrect Skin Tones",
        "Incorrect_reflection_mapping" : "Incorrect reflection mapping",
        "Incorrect_wheel_geometry" : "Incorrect wheel geometry",
        "Irregular_proportions_in_mechanical_components" : "Irregular proportions in mechanical components",
        "Jagged edges in curved structures" : "Jagged edges in curved structures",
        "Loss of fine detail in complex structures" : "Loss of fine detail in complex structures",
        "Metallic_surface_artifacts" : "Metallic surface artifacts",
        "Misaligned bilateral elements in animal faces" : "Misaligned bilateral elements in animal faces",
        "Misaligned body panels" : "Misaligned body panels",
        "Misshapen_Ears_or_Appendages" : "Misshapen ears or appendages",
        "Missing_ambient_occlusion" : "Missing ambient occlusion",
        "Movie-poster like composition of ordinary scenes" : "Movie-poster like composition of ordinary scenes",
        "Multiple_inconsistent_shadow_sources" : "Multiple inconsistent shadow sources",
        "Multiple_light_source_conflicts" : "Multiple light source conflicts",
        "Non_manifold_geometries_in_rigid_structures" : "Non-manifold geometries in rigid structures",
        "Over-sharpening artifacts" : "Over-sharpening artifacts",
        "Over-smoothing of natural textures" : "Over-smoothing of natural textures",
        "Physically_impossible_structural_elements" : "Physically impossible structural elements",
        "Random noise patterns in detailed areas" : "Random noise patterns in detailed areas",
        "Regular grid-like artifacts in textures" : "Regular grid-like artifacts in textures",
        "Repeated element patterns" : "Repeated element patterns",
        "Scale_inconsistencies_within_single_objects" : "Scale inconsistencies within single objects",
        "Scale_inconsistencies_within_the_same_object_class" : "Scale inconsistencies within the same object class",
        "Spatial_relationship_errors" : "Spatial relationship errors",
        "Synthetic material appearance" : "Synthetic material appearance",
        "Systematic color distribution anomalies" : "Systematic color distribution anomalies",
        "Texture bleeding between adjacent regions" : "Texture bleeding between adjacent regions",
        "Texture repetition patterns" : "Texture repetition patterns",
        "Unnatural color transitions" : "Unnatural color transitions",
        "Unnatural_Lighting_Gradients" : "Unnatural Lighting Gradients",
        "Unnatural_Pose_Artifacts" : "Unnatural pose artifacts",
        "Unrealistic_Eye_Reflections" : "Unrealistic eye reflections",
        "Unrealistic_specular_highlights" : "Unrealistic specular highlights",
        "excessive_sharpness" : "Excessive sharpness in certain image regions",
        "incorrect_perspective" : "Incorrect perspective rendering",
        "resolution_inconsistencies" : "Resolution inconsistencies within regions",
        "unnaturally_glossy_surfaces" : "Unnaturally glossy surfaces"
    }

    # Finding artifacts in image and generating explanations

    image = Image.open(image_path)
    artifact_list = result(image)
    artifact_name_found = []
    explanation_generated = []
    
    for artifact in artifact_list:
        artifact_exp_each = artifact_with_explanation(image_path,artifact)
        explanation_generated.append(artifact_exp_each)
        artifact_name_found.append(artifact)

    loc_list = []
    for artifact in artifact_name_found:
        image_with_bb, heatmap, contours = show(image_path,artifact)
        explanation_with_location = locating_artifact(
            image, image_with_bb, artifact, contours
        )
        loc_list.append(explanation_with_location)

    dict_artifacts_with_explanation = {}
    for j in range(len(artifact_name_found)):
        safe_key = keys.get(artifact_name_found[j], artifact_name_found[j])
        if loc_list[j]!="in the image":
            dict_artifacts_with_explanation[safe_key] = explanation_generated[j]+f" The artifact is present at the {loc_list[j]} of the image."
        else:
            dict_artifacts_with_explanation[safe_key] = explanation_generated[j]

    list_output = [{
            "index" : image_path,
            "explanation" : dict_artifacts_with_explanation
        }]

    return list_output

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference.py <path_to_image> <huggingface_token>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    hf_token = sys.argv[2]
    
    # Replace 'path/to/your/image.jpg' with actual path
    predicted_class = image_classification(input_image_path)

    print(f"The image is predicted to be: {predicted_class}")

    if predicted_class == "fake":
        print("Detecting Artifacts\n")
        explanation_output = artifact_detection(input_image_path, hf_token)
        print("Artifacts and explanation : ")
        print(explanation_output)
