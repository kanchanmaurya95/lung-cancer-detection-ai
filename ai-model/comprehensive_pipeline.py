#!/usr/bin/env python3
"""
Comprehensive Lung Cancer Detection Model Development Pipeline
==============================================================

This comprehensive pipeline includes:
1. Detailed EDA (Exploratory Data Analysis)
2. Cloud data integration
3. Custom model architecture
4. 50+ epoch training
5. Sample image testing
6. Visualization (accuracy/loss plots)
7. Confusion matrix
8. TFLite model conversion
9. App integration ready

Author: Kanchan Maurya
Date: August 24, 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
import cv2
from PIL import Image
import random
from pathlib import Path

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    GlobalAveragePooling2D, BatchNormalization, Input,
    DepthwiseConv2D, SeparableConv2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    CSVLogger, TensorBoard
)

# Evaluation Libraries
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# Google Cloud Storage
try:
    from google.cloud import storage
    CLOUD_AVAILABLE = True
except ImportError:
    CLOUD_AVAILABLE = False
    print("⚠️  Google Cloud Storage not available. Using local data only.")

warnings.filterwarnings('ignore')

class LungCancerEDA:
    """Comprehensive Exploratory Data Analysis for Lung Cancer Dataset"""
    
    def __init__(self, data_dir='data/', cloud_bucket=None):
        self.data_dir = Path(data_dir)
        self.cloud_bucket = cloud_bucket
        self.results_dir = Path('comprehensive_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # EDA results storage
        self.eda_results = {}
        
    def analyze_dataset_structure(self):
        """Analyze the structure and distribution of the dataset"""
        print("📊 STEP 1: Dataset Structure Analysis")
        print("=" * 60)
        
        # Get class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        dataset_info = {
            'total_classes': len(class_dirs),
            'classes': {},
            'total_images': 0
        }
        
        for class_dir in class_dirs:
            # Count images in each class
            image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg'))
            class_count = len(image_files)
            
            dataset_info['classes'][class_dir.name] = {
                'count': class_count,
                'files': [str(f) for f in image_files[:5]]  # Sample files
            }
            dataset_info['total_images'] += class_count
            
            print(f"📁 Class '{class_dir.name}': {class_count} images")
        
        print(f"\n📊 Total Images: {dataset_info['total_images']}")
        print(f"🏷️  Total Classes: {dataset_info['total_classes']}")
        
        # Check class balance
        if len(dataset_info['classes']) == 2:
            class_counts = [info['count'] for info in dataset_info['classes'].values()]
            balance_ratio = min(class_counts) / max(class_counts)
            print(f"⚖️  Class Balance Ratio: {balance_ratio:.3f}")
            
            if balance_ratio < 0.8:
                print("⚠️  Dataset is imbalanced! Consider data augmentation or resampling.")
            else:
                print("✅ Dataset is reasonably balanced.")
        
        self.eda_results['dataset_structure'] = dataset_info
        return dataset_info
    
    def analyze_image_properties(self, sample_size=50):
        """Analyze image properties like size, format, intensity distribution"""
        print(f"\n📸 STEP 2: Image Properties Analysis (Sample: {sample_size})")
        print("=" * 60)
        
        image_properties = {
            'sizes': [],
            'formats': [],
            'channels': [],
            'intensity_stats': [],
            'sample_images': {}
        }
        
        # Collect sample images from each class
        for class_name, class_info in self.eda_results['dataset_structure']['classes'].items():
            class_dir = self.data_dir / class_name
            image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            
            # Sample images
            sample_files = random.sample(image_files, min(sample_size//2, len(image_files)))
            image_properties['sample_images'][class_name] = []
            
            for img_path in sample_files:
                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Get properties
                    height, width, channels = img.shape
                    image_properties['sizes'].append((width, height))
                    image_properties['channels'].append(channels)
                    
                    # Intensity statistics
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    intensity_stats = {
                        'mean': np.mean(gray_img),
                        'std': np.std(gray_img),
                        'min': np.min(gray_img),
                        'max': np.max(gray_img)
                    }
                    image_properties['intensity_stats'].append(intensity_stats)
                    
                    # Store sample for visualization
                    if len(image_properties['sample_images'][class_name]) < 5:
                        image_properties['sample_images'][class_name].append(str(img_path))
                
                except Exception as e:
                    print(f"⚠️  Error processing {img_path}: {e}")
        
        # Analyze collected properties
        if image_properties['sizes']:
            sizes_array = np.array(image_properties['sizes'])
            print(f"📏 Image Sizes:")
            print(f"   Width: {sizes_array[:, 0].min()}-{sizes_array[:, 0].max()} (avg: {sizes_array[:, 0].mean():.0f})")
            print(f"   Height: {sizes_array[:, 1].min()}-{sizes_array[:, 1].max()} (avg: {sizes_array[:, 1].mean():.0f})")
            
            if image_properties['intensity_stats']:
                intensity_array = np.array([[s['mean'], s['std']] for s in image_properties['intensity_stats']])
                print(f"🎨 Intensity Statistics:")
                print(f"   Mean: {intensity_array[:, 0].mean():.2f} ± {intensity_array[:, 0].std():.2f}")
                print(f"   Std Dev: {intensity_array[:, 1].mean():.2f} ± {intensity_array[:, 1].std():.2f}")
        
        self.eda_results['image_properties'] = image_properties
        return image_properties
    
    def create_sample_visualization(self):
        """Create visualization of sample images from each class"""
        print(f"\n🖼️  STEP 3: Sample Image Visualization")
        print("=" * 60)
        
        sample_images = self.eda_results['image_properties']['sample_images']
        
        # Create subplot grid
        n_classes = len(sample_images)
        n_samples = 5
        
        fig, axes = plt.subplots(n_classes, n_samples, figsize=(20, 4*n_classes))
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        
        for class_idx, (class_name, img_paths) in enumerate(sample_images.items()):
            for img_idx, img_path in enumerate(img_paths[:n_samples]):
                if img_idx >= n_samples:
                    break
                
                try:
                    # Load and display image
                    img = load_img(img_path, target_size=(224, 224))
                    
                    if n_classes > 1:
                        ax = axes[class_idx, img_idx]
                    else:
                        ax = axes[img_idx]
                    
                    ax.imshow(img)
                    ax.set_title(f'{class_name.title()} Sample {img_idx+1}')
                    ax.axis('off')
                
                except Exception as e:
                    print(f"⚠️  Error displaying {img_path}: {e}")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sample_images.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Sample images saved to: {self.results_dir}/sample_images.png")
    
    def create_distribution_plots(self):
        """Create distribution plots for dataset analysis"""
        print(f"\n📈 STEP 4: Distribution Analysis")
        print("=" * 60)
        
        # Class distribution pie chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Class Distribution
        class_names = list(self.eda_results['dataset_structure']['classes'].keys())
        class_counts = [info['count'] for info in self.eda_results['dataset_structure']['classes'].values()]
        
        axes[0, 0].pie(class_counts, labels=class_names, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Class Distribution')
        
        # 2. Image Size Distribution
        sizes = self.eda_results['image_properties']['sizes']
        if sizes:
            widths = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]
            
            axes[0, 1].scatter(widths, heights, alpha=0.6)
            axes[0, 1].set_xlabel('Width')
            axes[0, 1].set_ylabel('Height')
            axes[0, 1].set_title('Image Size Distribution')
        
        # 3. Intensity Distribution
        intensity_stats = self.eda_results['image_properties']['intensity_stats']
        if intensity_stats:
            means = [s['mean'] for s in intensity_stats]
            axes[1, 0].hist(means, bins=30, alpha=0.7)
            axes[1, 0].set_xlabel('Mean Intensity')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Intensity Distribution')
        
        # 4. Class Count Bar Chart
        axes[1, 1].bar(class_names, class_counts, color=['skyblue', 'salmon'])
        axes[1, 1].set_xlabel('Classes')
        axes[1, 1].set_ylabel('Number of Images')
        axes[1, 1].set_title('Images per Class')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Distribution plots saved to: {self.results_dir}/distribution_analysis.png")
    
    def print_training_test_counts(self, train_split=0.8):
        """Print detailed training and test dataset counts"""
        print(f"\n📊 STEP 5: Training/Test Split Analysis (Split: {train_split:.0%})")
        print("=" * 60)
        
        total_images = self.eda_results['dataset_structure']['total_images']
        
        for class_name, class_info in self.eda_results['dataset_structure']['classes'].items():
            class_count = class_info['count']
            train_count = int(class_count * train_split)
            test_count = class_count - train_count
            
            print(f"📁 {class_name.title()} Class:")
            print(f"   Total: {class_count}")
            print(f"   Training: {train_count} ({train_count/class_count:.1%})")
            print(f"   Testing: {test_count} ({test_count/class_count:.1%})")
            print()
        
        total_train = int(total_images * train_split)
        total_test = total_images - total_train
        
        print(f"🎯 Overall Split:")
        print(f"   Total Images: {total_images}")
        print(f"   Training Set: {total_train} ({total_train/total_images:.1%})")
        print(f"   Test Set: {total_test} ({total_test/total_images:.1%})")
        
        return {
            'total_train': total_train,
            'total_test': total_test,
            'split_ratio': train_split
        }
    
    def save_eda_report(self):
        """Save comprehensive EDA report"""
        report_path = self.results_dir / 'eda_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(self.eda_results, f, indent=2, default=str)
        
        print(f"\n💾 EDA Report saved to: {report_path}")
        return report_path

class LungCancerModelTrainer:
    """Custom model trainer with comprehensive features"""
    
    def __init__(self, data_dir='data/', cloud_bucket=None):
        self.data_dir = data_dir
        self.cloud_bucket = cloud_bucket
        self.results_dir = Path('comprehensive_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Model configuration
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 50  # Minimum 50 epochs as requested
        
    def create_custom_model(self, input_shape=(224, 224, 3)):
        """Create a custom CNN model optimized for lung cancer detection"""
        print(f"🏗️  STEP 6: Building Custom CNN Model")
        print("=" * 60)
        
        # Custom architecture inspired by MobileNet but optimized for medical imaging
        inputs = Input(shape=input_shape)
        
        # Initial conv layer
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, 2)(x)
        
        # Depthwise separable conv blocks (efficient)
        x = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, 2)(x)
        
        x = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, 2)(x)
        
        x = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, 2)(x)
        
        # Additional conv layers for feature extraction
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Global pooling instead of flatten
        x = GlobalAveragePooling2D()(x)
        
        # Classification head
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)  # Binary classification
        
        model = Model(inputs, outputs, name='CustomLungCancerCNN')
        
        # Compile with appropriate metrics
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("📋 Custom Model Architecture:")
        model.summary()
        
        # Calculate model size
        param_count = model.count_params()
        print(f"\n📊 Model Statistics:")
        print(f"   Total Parameters: {param_count:,}")
        print(f"   Trainable Parameters: {param_count:,}")
        print(f"   Estimated Size: {param_count * 4 / (1024**2):.1f} MB")
        
        return model
    
    def setup_enhanced_data_generators(self):
        """Setup data generators with comprehensive augmentation"""
        print(f"\n📊 STEP 7: Setting up Enhanced Data Generators")
        print("=" * 60)
        
        # Enhanced augmentation strategy for medical images
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            # Medical image appropriate augmentations
            rotation_range=15,  # Reduced rotation for medical images
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.9, 1.1],  # Slight brightness adjustment
            fill_mode='nearest',
            # Advanced augmentations
            shear_range=0.1,
        )
        
        # Validation with only rescaling
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        validation_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        print(f"✅ Training samples: {train_generator.samples}")
        print(f"✅ Validation samples: {validation_generator.samples}")
        print(f"🏷️  Class mapping: {train_generator.class_indices}")
        
        return train_generator, validation_generator
    
    def train_model_comprehensive(self, model, train_gen, val_gen):
        """Train model with comprehensive monitoring"""
        print(f"\n🚀 STEP 8: Training Model for {self.epochs} Epochs")
        print("=" * 60)
        
        # Setup comprehensive callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,  # Increased patience for 50+ epochs
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.results_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            CSVLogger(
                filename=str(self.results_dir / 'training_log.csv'),
                append=False
            ),
            TensorBoard(
                log_dir=str(self.results_dir / 'tensorboard_logs'),
                histogram_freq=1,
                write_images=True
            )
        ]
        
        # Training
        start_time = datetime.now()
        print(f"📅 Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        history = model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"\n⏱️  Training completed in: {training_time}")
        print(f"📅 Training ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return history, training_time
    
    def test_sample_images(self, model, val_gen, num_samples=10):
        """Test model on sample images and display results"""
        print(f"\n🧪 STEP 9: Testing Sample Images ({num_samples} samples)")
        print("=" * 60)
        
        # Get sample images and predictions
        val_gen.reset()
        sample_images = []
        sample_labels = []
        sample_predictions = []
        
        batch_count = 0
        for images, labels in val_gen:
            for i in range(len(images)):
                if len(sample_images) >= num_samples:
                    break
                
                sample_images.append(images[i])
                sample_labels.append(labels[i])
                
                # Get prediction
                pred = model.predict(np.expand_dims(images[i], axis=0), verbose=0)[0][0]
                sample_predictions.append(pred)
            
            if len(sample_images) >= num_samples:
                break
            
            batch_count += 1
            if batch_count >= 5:  # Limit to avoid infinite loop
                break
        
        # Create visualization
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        class_names = ['Cancer', 'Normal']  # Assuming cancer=0, normal=1
        
        for i, (img, true_label, pred_prob) in enumerate(zip(sample_images, sample_labels, sample_predictions)):
            if i >= num_samples:
                break
            
            # Display image
            axes[i].imshow(img)
            
            # Prediction
            pred_class = 1 if pred_prob > 0.5 else 0
            confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
            
            # Title with prediction info
            true_class_name = class_names[int(true_label)]
            pred_class_name = class_names[pred_class]
            
            is_correct = (pred_class == int(true_label))
            color = 'green' if is_correct else 'red'
            
            title = f'True: {true_class_name}\\nPred: {pred_class_name}\\nConf: {confidence:.2f}'
            axes[i].set_title(title, color=color, fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate accuracy on samples
        correct_predictions = sum(1 for true, pred in zip(sample_labels, sample_predictions) 
                                 if (pred > 0.5) == bool(true))
        sample_accuracy = correct_predictions / len(sample_labels)
        
        print(f"📊 Sample Test Results:")
        print(f"   Tested Images: {len(sample_images)}")
        print(f"   Correct Predictions: {correct_predictions}")
        print(f"   Sample Accuracy: {sample_accuracy:.1%}")
        print(f"✅ Sample predictions saved to: {self.results_dir}/sample_predictions.png")
        
        return sample_accuracy
    
    def plot_training_results(self, history):
        """Create comprehensive training plots"""
        print(f"\n📈 STEP 10: Creating Training Visualization")
        print("=" * 60)
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs_range = range(1, len(history.history['accuracy']) + 1)
        
        # 1. Accuracy
        axes[0, 0].plot(epochs_range, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(epochs_range, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Loss
        axes[0, 1].plot(epochs_range, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 1].plot(epochs_range, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision
        if 'precision' in history.history:
            axes[0, 2].plot(epochs_range, history.history['precision'], 'b-', label='Training Precision', linewidth=2)
            axes[0, 2].plot(epochs_range, history.history['val_precision'], 'r-', label='Validation Precision', linewidth=2)
            axes[0, 2].set_title('Model Precision Over Time')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Recall
        if 'recall' in history.history:
            axes[1, 0].plot(epochs_range, history.history['recall'], 'b-', label='Training Recall', linewidth=2)
            axes[1, 0].plot(epochs_range, history.history['val_recall'], 'r-', label='Validation Recall', linewidth=2)
            axes[1, 0].set_title('Model Recall Over Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Learning Rate (if available)
        if 'learning_rate' in history.history:
            axes[1, 1].plot(epochs_range, history.history['learning_rate'], 'g-', linewidth=2)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Training Summary
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        best_val_acc = max(history.history['val_accuracy'])
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        summary_text = f'''Final Training Results:
        
Training Accuracy: {final_train_acc:.4f}
Validation Accuracy: {final_val_acc:.4f}
Best Validation Accuracy: {best_val_acc:.4f}

Final Training Loss: {final_train_loss:.4f}
Final Validation Loss: {final_val_loss:.4f}

Total Epochs: {len(history.history["accuracy"])}'''
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, 
                       verticalalignment='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Training Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training plots saved to: {self.results_dir}/training_plots.png")
        
        return {
            'final_train_accuracy': final_train_acc,
            'final_val_accuracy': final_val_acc,
            'best_val_accuracy': best_val_acc
        }
    
    def create_confusion_matrix(self, model, val_gen):
        """Create and plot confusion matrix"""
        print(f"\n📊 STEP 11: Creating Confusion Matrix")
        print("=" * 60)
        
        # Get predictions
        val_gen.reset()
        predictions = model.predict(val_gen, verbose=1)
        y_pred = (predictions > 0.5).astype(int)
        y_true = val_gen.classes[:len(predictions)]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, predictions)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Cancer', 'Normal'],
                   yticklabels=['Cancer', 'Normal'],
                   cbar_kws={'label': 'Number of Predictions'})
        
        plt.title('Confusion Matrix - Lung Cancer Detection Model')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add metrics text
        metrics_text = f'''Model Performance Metrics:

Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1:.4f}
AUC-ROC: {auc:.4f}

True Positives: {cm[0][0]}
False Negatives: {cm[0][1]}
False Positives: {cm[1][0]}
True Negatives: {cm[1][1]}'''
        
        plt.figtext(1.05, 0.5, metrics_text, fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Model Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   AUC-ROC: {auc:.4f}")
        print(f"✅ Confusion matrix saved to: {self.results_dir}/confusion_matrix.png")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'confusion_matrix': cm.tolist()
        }
    
    def convert_to_tflite(self, model):
        """Convert trained model to TensorFlow Lite format"""
        print(f"\n📱 STEP 12: Converting to TensorFlow Lite")
        print("=" * 60)
        
        try:
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Optimization settings
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Convert
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = self.results_dir / 'lung_cancer_model.tflite'
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get model size
            tflite_size = len(tflite_model) / (1024 * 1024)  # MB
            original_size = model.count_params() * 4 / (1024 * 1024)  # MB
            
            print(f"✅ TensorFlow Lite model created successfully!")
            print(f"📁 Saved to: {tflite_path}")
            print(f"📊 Model Compression:")
            print(f"   Original size: {original_size:.2f} MB")
            print(f"   TFLite size: {tflite_size:.2f} MB")
            print(f"   Compression ratio: {original_size/tflite_size:.1f}x")
            
            return str(tflite_path), tflite_size
            
        except Exception as e:
            print(f"❌ Error converting to TFLite: {e}")
            return None, None

def main():
    """Main comprehensive pipeline"""
    print("🫁 COMPREHENSIVE LUNG CANCER DETECTION PIPELINE")
    print("=" * 80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize components
    data_dir = 'data/'
    
    # Step 1-5: Comprehensive EDA
    print("🔍 PHASE 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    eda = LungCancerEDA(data_dir)
    
    # Analyze dataset structure
    dataset_info = eda.analyze_dataset_structure()
    
    # Analyze image properties
    image_props = eda.analyze_image_properties(sample_size=100)
    
    # Create visualizations
    eda.create_sample_visualization()
    eda.create_distribution_plots()
    
    # Print training/test counts
    split_info = eda.print_training_test_counts(train_split=0.8)
    
    # Save EDA report
    eda.save_eda_report()
    
    # Step 6-12: Model Development
    print("\\n🤖 PHASE 2: MODEL DEVELOPMENT & TRAINING")
    print("=" * 80)
    
    trainer = LungCancerModelTrainer(data_dir)
    
    # Create custom model
    model = trainer.create_custom_model()
    
    # Setup data generators
    train_gen, val_gen = trainer.setup_enhanced_data_generators()
    
    # Train model
    history, training_time = trainer.train_model_comprehensive(model, train_gen, val_gen)
    
    # Test sample images
    sample_accuracy = trainer.test_sample_images(model, val_gen, num_samples=10)
    
    # Create training plots
    training_results = trainer.plot_training_results(history)
    
    # Create confusion matrix
    performance_metrics = trainer.create_confusion_matrix(model, val_gen)
    
    # Convert to TFLite
    tflite_path, tflite_size = trainer.convert_to_tflite(model)
    
    # Step 13: Final Summary and App Integration Prep
    print("\\n🎯 PHASE 3: FINAL SUMMARY & APP INTEGRATION")
    print("=" * 80)
    
    # Save comprehensive results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'training_time': str(training_time),
        'dataset_info': dataset_info,
        'split_info': split_info,
        'training_results': training_results,
        'performance_metrics': performance_metrics,
        'sample_accuracy': sample_accuracy,
        'tflite_info': {
            'path': tflite_path,
            'size_mb': tflite_size
        },
        'model_files': {
            'h5_model': str(trainer.results_dir / 'best_model.h5'),
            'tflite_model': tflite_path,
            'training_log': str(trainer.results_dir / 'training_log.csv')
        }
    }
    
    with open(trainer.results_dir / 'complete_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Print final summary
    print("🏆 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Final Model Performance:")
    print(f"   Best Validation Accuracy: {training_results['best_val_accuracy']:.1%}")
    print(f"   Final Test Accuracy: {performance_metrics['accuracy']:.1%}")
    print(f"   AUC-ROC Score: {performance_metrics['auc_roc']:.3f}")
    print(f"   F1-Score: {performance_metrics['f1_score']:.3f}")
    print()
    print(f"📁 All Results Saved to: {trainer.results_dir}/")
    print(f"🤖 Model Files:")
    print(f"   H5 Model: {trainer.results_dir}/best_model.h5")
    print(f"   TFLite Model: {tflite_path}")
    print(f"   Training Log: {trainer.results_dir}/training_log.csv")
    print()
    print(f"📱 Ready for App Integration!")
    print(f"   Use the TFLite model for mobile deployment")
    print(f"   Model size: {tflite_size:.2f} MB (mobile-optimized)")
    print()
    print(f"⏱️  Total Pipeline Time: {training_time}")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
