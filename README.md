# Image Classification and Artifact Identification

## Overview
Advanced generative models like Stable Diffusion and Adobe Firefly often introduce subtle artifacts that differentiate AI-generated images from real ones. This project aims to develop a machine learning solution to address the growing challenges of detecting and explaining these AI-generated images. 

The solution includes two primary tasks:
- **Task 1: Detection of AI-Generated Images:** Classify images as real or AI-generated.
- **Task 2: Artifact Identification & Explanation:** Identify distinguishing artifacts in AI-generated images and provide concise, human-interpretable explanations.

This approach enhances transparency, aids in combating misinformation, and builds trust in AI-driven content verification systems while maintaining computational efficiency.

## Objectives

### Task 1: Detection of AI-Generated Images
- Classifying images into real or AI-generated.
- **Results:**
  - F1 Score: `95.56%`
  - Recall: `97.93%`
  - Precision: `93.31%`

### Task 2: Artifact Identification & Explanation
- Identify distinguishing artifacts in synthetic images and provide explanations to support classification decisions.
- **Evaluation:**
  - Intersection over Union (IoU) for artifact selection.
  - Content similarity with human-annotated explanations, evaluated using GPT-4.

## Our Approach
Our solution integrates lightweight classification with robust artifact identification and explanation generation:

### Task 1: Image Classification
- **Model:** Fine-tuned EfficientNet-V2B0 for high accuracy and low latency.
- **Training:** Adversarial training on the CIFAKE dataset with augmented perturbations.
- **Optimization:** A compact architecture ensures real-time, scalable detection.

![task_1_pipeline](task1_training_pipeline.png "Task_1_pipeline")

### Task 2: Artifact Identification & Explanation
- **Feature Extraction:** A pre-trained Vision Transformer (ViT) extracts fine-grained image features.
- **Artifact Detection:** Ensemble models (XGBoost, Random Forest) analyze the ViT features to detect inconsistencies like unnatural textures and lighting.
- **Explanation Generation:** The LLaVA model generates concise, localized explanations describing the artifact's characteristics. 
- **Localization:** SHAP-based heatmaps visualize the artifact locations, and bounding boxes provide precise localization. The results are structured in a JSON format for clarity.

![Pipeline](final_pipeline.png "Pipeline")
![Task 2 Pipeline](task2_training_pipeline.png "Task 2 Pipeline")

## File Structure

Your directory should look like this after extracting the files:
```text
Codebase/
├── Task_1/
│   ├── data/
│   │   ├── test/
│   │   └── train/
│   ├── inference_script.py
│   ├── EfficientNet_model.pkl
│   └── EfficientNet_training_script.ipynb
├── Task_2/
│   ├── data/
│   │   └── artifacts/
│   │       ├── Abruptly cut-off objects/
│   │       ├── Aliasing along high-contrast edges/
│   │       └── and 68 more...
│   ├── models/
│   │   ├── rf_models-20241126T171044Z-001/
│   │   ├── xgb_model_params/
│   │   └── xgb_models-no_params/
│   ├── training_pipelines/
│   │   ├── rf_training/
│   │   ├── XGB_no_params_training/
│   │   └── XGB_params_training/
│   ├── ai-image-artifacts.txt
│   ├── datasetGenerator.ipynb
│   └── inference_script_task2.py
├── CreatingPerturbations.py
└── inference.py

```

* `Task_1/`: Contains datasets, models, and training scripts for the classification task.
* `Task_2/`: Contains datasets, trained ensemble models, pipelines, and scripts for artifact detection and explanation.
* `inference.py`: The unified Python script to run both tasks simultaneously from the terminal.
* `CreatingPerturbations.py`: Python script used to create adversarially perturbed images.

## Steps to Execute the Pipeline

### 1. Install Dependencies

Ensure the required libraries are installed. Run the following command in your terminal:

```bash
pip install -r requirements.txt

```

*Alternatively:*

```bash
pip install transformers torch torchvision shap xgboost scikit-image matplotlib pillow opencv-python huggingface_hub tensorflow keras

```

### 2. Set Up the Environment

* Ensure GPU availability for faster processing (e.g., enable GPU support if running in Google Colab).
* **Hugging Face Token:** You will need a valid Hugging Face API token to download and run the LLaVA model.

### 3. Prepare Models and Files

Ensure all pre-trained models are correctly placed inside the `Codebase` directory:

* `Codebase/Task_1/EfficientNet_model.pkl`
* `Codebase/Task_2/models/xgb_model_params/xgb_models/`
* `Codebase/Task_2/models/xgb_models-no_params/xgb_models/`
* `Codebase/Task_2/models/rf_models-20241126T171044Z-001/rf_models/`

### 4. Run the Script

You can now run the entire pipeline directly from the terminal by passing the image path and your Hugging Face token as arguments:

```bash
python inference.py <path_to_your_image.jpg> <your_huggingface_token>

```

*Example:*

```bash
python inference.py sample_images/test_fake.png hf_abc123XYZ...

```

## Example Output

Below is an example of a heatmap and bounding box generated by the Task 2 pipeline to localize detected artifacts:

## Troubleshooting

* **Model Loading Errors:** Verify that all model `.pkl` and `.pth` files are correctly placed in the designated `Task_1` and `Task_2/models/` directories.
* **Missing Arguments:** If you get a `Usage:` error, ensure you are providing both the image path and the Hugging Face token in the terminal command.
* **Visualization Issues:** Ensure `opencv-python` and `matplotlib` are properly installed in your environment.
* **Out of Memory (OOM) Errors:** LLaVA and ViT models are resource-intensive. Ensure your machine/GPU has adequate VRAM, or run the script in a cloud environment like Colab.

