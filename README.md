# Dog Breed Image Classifier Project

This project is designed to classify dog breeds from images using a pre-trained deep learning model. It includes scripts for training, evaluating, and predicting using models like AlexNet, ResNet, and VGG. Additionally, the project compares performance across different models on various datasets.

## Project Overview

The project aims to:
- Classify images of pets (dogs) using deep learning.
- Use transfer learning by leveraging pre-trained models from `torchvision.models`.
- Provide a command-line interface for flexible model usage.

## Installation

### Dependencies
To run this project, the following Python libraries are required:
- `torch`
- `argparse`
- `PIL`
- `os`
- `json`
- `shutil`

Install dependencies with:
```bash
pip install torch pillow
```

## Project Files

### Core Scripts
- `check_images.py`: Verifies image file structure and formats.
- `classify_images.py`: Classifies images using the chosen deep learning model.
- `get_input_args.py`: Parses command-line arguments, including model architecture and file paths.
- `get_pet_labels.py`: Extracts labels (breeds) from image filenames.
- `adjust_results4_isadog.py`: Adjusts classification results to account for whether the image is a dog.
- `calculates_results_stats.py`: Computes statistics (accuracy, precision, recall, etc.) based on model predictions.
- `print_results.py`: Prints out a summary of classification results and statistics.
- `print_functions_for_lab_checks.py`: Helper functions for debugging and validation.

### Additional Files
- `run_models_batch.sh`: Batch script to run model training and classification for pet images.
- `run_models_batch_uploaded.sh`: Batch script for running models on uploaded images.
- `imagenet1000_clsid_to_human.txt`: Maps ImageNet class IDs to human-readable labels.
- `dognames.txt`: Contains a list of dog breeds recognized by the classifier.

### Pretrained Model Results
- `alexnet_pet-images.txt`: Classification results using the AlexNet model on pet images.
- `resnet_pet-images.txt`: Classification results using the ResNet model on pet images.
- `vgg_pet-images.txt`: Classification results using the VGG model on pet images.
- `alexnet_uploaded-images.txt`: Classification results using the AlexNet model on uploaded images.
- `resnet_uploaded-images.txt`: Classification results using the ResNet model on uploaded images.
- `vgg_uploaded-images.txt`: Classification results using the VGG model on uploaded images.

### Folders
- `pet_images/`: Contains the pet images for model training and testing.
- `uploaded_images/`: Contains the uploaded images used for classification testing.

## Usage

### Command Line Arguments
You can run the scripts from the command line, with the following customizable arguments:

1. **Classify Images**
   ```bash
   python classify_images.py --dir <path_to_images> --arch <model_architecture> --dogfile dognames.txt
   ```

   Example:
   ```bash
   python classify_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
   ```

2. **Check Images**
   ```bash
   python check_images.py --dir <path_to_images>
   ```

3. **Run Models Batch**
   ```bash
   ./run_models_batch.sh
   ```

### Options for Command Line:
- `--dir`: Directory path to the images.
- `--arch`: Model architecture to use (`alexnet`, `resnet`, `vgg`).
- `--dogfile`: Text file containing a list of recognized dog breeds.

## Results

This project tests the accuracy of different pre-trained models on pet images, specifically dogs. It evaluates the model's ability to correctly classify whether the image is a dog, and if so, identifies the breed.

Key metrics such as precision, recall, and F1 score are calculated and printed using the `print_results.py` script.
