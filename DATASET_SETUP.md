# 📊 Dataset Setup Instructions

Your GitHub repository now contains the clean AI training pipeline, but the dataset images are not included due to size constraints.

## 🗂️ Setting Up Your Dataset

To use this training pipeline, you need to add your lung cancer dataset:

### 1. Create Dataset Structure
```bash
cd ai-model/data/
mkdir -p cancer normal
```

### 2. Add Your Images
- Place lung cancer X-ray images in `ai-model/data/cancer/`
- Place normal lung X-ray images in `ai-model/data/normal/`

### 3. Supported Formats
- PNG, JPG, JPEG images
- Recommended size: 224x224 pixels or larger
- Images will be automatically resized during training

## 🚀 Start Training

Once your dataset is ready:
```bash
cd ai-model
python3 comprehensive_pipeline.py
```

## 📝 Notes

- The repository includes sample dataset info files for reference
- Training results and models will be saved in `comprehensive_results/`
- Large model files (`.h5`) are excluded from git to save space
- TensorFlow Lite models (`.tflite`) are included as they're smaller

Happy training! 🔬
