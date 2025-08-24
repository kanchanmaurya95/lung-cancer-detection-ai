# 🔬 Lung Cancer Detection - AI Training

Complete AI training pipeline for lung cancer detection using deep learning.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
cd ai-model
python3 comprehensive_pipeline.py
```

## 📁 Structure

```
ai-model/
├── comprehensive_pipeline.py    # Main training script
├── complete_system.py          # TensorFlow Lite conversion  
├── data/                       # Dataset (cancer/, normal/)
└── comprehensive_results/      # Training outputs & models
```

## � Features

- **EDA Analysis** - Dataset distribution and quality assessment
- **Custom CNN** - Separable convolutions optimized for medical imaging
- **Training Pipeline** - Data augmentation, early stopping, checkpoints
- **Model Evaluation** - Confusion matrix, performance metrics
- **TensorFlow Lite** - Mobile-optimized model conversion

## 📂 Outputs

Training generates in `comprehensive_results/`:
- `best_model.h5` - Trained Keras model
- `lung_cancer_model.tflite` - TensorFlow Lite model
- `distribution_analysis.png` - Dataset analysis plots
- `sample_images.png` - Sample predictions
- `training_log.csv` - Training metrics

## ⚠️ Disclaimer

Educational purposes only. Not for medical diagnosis.

Start training: `cd ai-model && python3 comprehensive_pipeline.py` 🚀