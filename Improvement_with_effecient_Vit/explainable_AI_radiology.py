# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import cv2
from glob import glob
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# %% [markdown]
# 
# # Chest X-ray Dataset
# 

# %%

base_path = '/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database'
normal_path = os.path.join(base_path, 'Normal')
tb_path = os.path.join(base_path, 'Tuberculosis')

def create_dataset_df(normal_dir, tb_dir):
    normal_files = glob(os.path.join(normal_dir, '*.png'))
    tb_files = glob(os.path.join(tb_dir, '*.png'))
    
    files = normal_files + tb_files
    labels = ['Normal'] * len(normal_files) + ['Tuberculosis'] * len(tb_files)
    
    return pd.DataFrame({
        'image_path': files,
        'label': labels
    })


if os.path.exists(normal_path) and os.path.exists(tb_path):
    df = create_dataset_df(normal_path, tb_path)
    
    print("\n Dataset Information:")
    print(f"Total number of images: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    plt.figure(figsize=(10, 6))
    df['label'].value_counts().plot(kind='bar')
    plt.title('Class Distribution in TB Chest X-ray Dataset')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    plt.figure(figsize=(15, 10))
    for i, label in enumerate(['Normal', 'Tuberculosis']):
        sample_paths = df[df['label'] == label]['image_path'].sample(min(3, len(df[df['label'] == label]))).tolist()
        
        for j, path in enumerate(sample_paths):
            plt.subplot(2, 3, i*3 + j + 1)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(f"{label} (Shape: {img.shape})")
            plt.axis('off')
    plt.tight_layout()
    plt.show()
    
   
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']  
    )
    
    print("\n Train/Test Split:")
    print(f"Training set: {len(train_df)} images")
    print(f"Testing set: {len(test_df)} images")
    print("\nClass distribution in training set:")
    print(train_df['label'].value_counts())
    print("\nClass distribution in testing set:")
    print(test_df['label'].value_counts())
    
    # Image statistics analysis
    def analyze_image_statistics(dataframe):
        heights = []
        widths = []
        aspects = []
        
        sample_size = min(100, len(dataframe))
        sample_df = dataframe.sample(sample_size)
        print(f"Analyzing {sample_size} sample images...")
        for path in sample_df['image_path']:
            try:
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    heights.append(h)
                    widths.append(w)
                    aspects.append(w/h)
            except Exception as e:
                print(f"Error processing image {path}: {e}")
        
        return {
            'height': {'mean': np.mean(heights), 'std': np.std(heights)},
            'width': {'mean': np.mean(widths), 'std': np.std(widths)},
            'aspect_ratio': {'mean': np.mean(aspects), 'std': np.std(aspects)}
        }
    
    # Analyze image statistics
    stats = analyze_image_statistics(df)
    print("\n Image Statistics:")
    print(f"Height (pixels): Mean={stats['height']['mean']:.1f}, Std={stats['height']['std']:.1f}")
    print(f"Width (pixels): Mean={stats['width']['mean']:.1f}, Std={stats['width']['std']:.1f}")
    print(f"Aspect Ratio: Mean={stats['aspect_ratio']['mean']:.3f}")
    
    # Class-specific image analysis
    def analyze_class_features(label, num_samples=3):
        class_df = df[df['label'] == label]
        sample_paths = class_df['image_path'].sample(num_samples).tolist()
        plt.figure(figsize=(15, 12))
        for i, path in enumerate(sample_paths):
            plt.subplot(num_samples, 3, i*3 + 1)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(f"Original")
            plt.axis('off')
            plt.subplot(num_samples, 3, i*3 + 2)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            plt.hist(gray.ravel(), bins=50, color='gray')
            plt.title(f"Intensity Distribution")
            plt.subplot(num_samples, 3, i*3 + 3)
            edges = cv2.Canny(gray, 100, 200)
            plt.imshow(edges, cmap='gray')
            plt.title(f"Edge Detection")
            plt.axis('off')
            
        plt.suptitle(f"{label} Class Analysis", fontsize=16, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    print("\n Analyzing class features...")
    analyze_class_features('Normal')
    analyze_class_features('Tuberculosis')
    def preprocess_for_model(image_path, target_size=(224, 224)):
        """Preprocess an image for model training"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0 
        return img
    
    #  preprocessing
    sample_img_path = df['image_path'].iloc[0]
    original_img = cv2.imread(sample_img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    processed_img = preprocess_for_model(sample_img_path)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title(f"Original ({original_img.shape})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed_img)
    plt.title(f"Processed ({processed_img.shape})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

# %%
!pip install transformers timm torch torchvision scikit-learn opencv-python matplotlib pandas numpy tqdm


# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from transformers import AutoModel
import time
import gc
import traceback
from PIL import Image

# to avoid memory issues
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print_gpu_memory()


def load_segmentation_model():
    print("Loading pretrained chest X-ray segmentation model...")
    try:
        model = AutoModel.from_pretrained("ianpan/chest-x-ray-basic", trust_remote_code=True)
        model = model.eval().to(device)
        print("Successfully loaded pretrained segmentation model")
        return model
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        print("Using fallback segmentation method")
        return None

# if unet doesnt work so manual
def basic_lung_segmentation(image):
    """
    Basic lung segmentation using image processing as fallback
    """
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    
    _, binary = cv2.threshold(enhanced, 90, 255, cv2.THRESH_BINARY)
    
    
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to original image
    if len(image.shape) == 3:
        mask = np.stack([binary, binary, binary], axis=2) / 255.0
    else:
        mask = binary / 255.0
    
    segmented = image * mask
    
    return segmented


def segment_lungs(image, model=None):
    """
    Segment lungs from chest X-ray using the pretrained model
    """
    if model is None:
        return basic_lung_segmentation(image)
    
    orig_h, orig_w = image.shape[:2]
    
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = image
    
    try:
       
        x = model.preprocess(gray_img)  
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  
        x = x.float().to(device)
        
        # Generate prediction
        with torch.no_grad():
            out = model(x)
            torch.cuda.synchronize()  
        mask = out["mask"].squeeze().cpu().numpy()
        mask = mask.argmax(0)
        binary_mask = np.zeros_like(mask)
        binary_mask[mask == 1] = 1  # Right lung
        binary_mask[mask == 2] = 1  # Left lung
        if binary_mask.shape[0] != orig_h or binary_mask.shape[1] != orig_w:
            binary_mask = cv2.resize(binary_mask.astype(np.float32), (orig_w, orig_h))
        if len(image.shape) == 3:
            segmented = image * np.expand_dims(binary_mask, axis=2)
        else:
            segmented = image * binary_mask
        if segmented.dtype == np.float32 or segmented.dtype == np.float64: 
            segmented = np.clip(segmented, 0, 1)
            segmented = (segmented * 255).astype(np.uint8)
        return segmented
    except Exception as e:
        print(f"Error in segmentation: {e}")
        print(traceback.format_exc())
        return basic_lung_segmentation(image)


def preprocess_and_cache_segmented_images(df, model, cache_dir='segmented_cache'):
    os.makedirs(cache_dir, exist_ok=True)
    print("Pre-processing and caching segmented images...")
    processed_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):   
        src_path = row['image_path']
        filename = os.path.basename(src_path)
        dest_path = os.path.join(cache_dir, filename)
        if os.path.exists(dest_path):
            processed_count += 1
            continue
        try:
            img = cv2.imread(src_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            segmented = segment_lungs(img, model)
            segmented = cv2.resize(segmented, (224, 224))
            cv2.imwrite(dest_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
            processed_count += 1
            if idx % 100 == 0:
                torch.cuda.empty_cache()       
        except Exception as e:
            print(f"Error processing {src_path}: {e}")
    
    print(f"Processed {processed_count} images. Cached at {cache_dir}/")
    new_df = df.copy()
    new_df['segmented_path'] = new_df['image_path'].apply(
        lambda x: os.path.join(cache_dir, os.path.basename(x))
    )
    return new_df


class OptimizedChestXRayDataset(Dataset):
    def __init__(self, dataframe, transform=None, use_segmented_paths=True):
        self.dataframe = dataframe
        self.transform = transform
        self.use_segmented_paths = use_segmented_paths and 'segmented_path' in dataframe.columns
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        
        if self.use_segmented_paths:
            img_path = self.dataframe.iloc[idx]['segmented_path']
        else:
            img_path = self.dataframe.iloc[idx]['image_path']
            
        label = 1 if self.dataframe.iloc[idx]['label'] == 'Tuberculosis' else 0
        
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224))
        
        
        if self.transform:
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
            
            img = self.transform(img)
        
        return img, label


def get_transforms(is_training=True):
    """Get image transformations"""
    if is_training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_optimized_dataloaders(train_df, test_df, batch_size=8):
    """Create data loaders with optimized settings for Kaggle P100"""
    train_transform = get_transforms(is_training=True)
    test_transform = get_transforms(is_training=False)
    
    train_dataset = OptimizedChestXRayDataset(
        train_df, 
        transform=train_transform,
        use_segmented_paths=True
    )
    
    test_dataset = OptimizedChestXRayDataset(
        test_df, 
        transform=test_transform,
        use_segmented_paths=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  
        pin_memory=True  
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  
        pin_memory=True  
    )
    
    return train_loader, test_loader


def setup_densenet201(num_classes=2):
    """Load and configure DenseNet201 for transfer learning"""
    
    model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    
    
    for param in model.parameters():
        param.requires_grad = False
    
    
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=15, use_early_stopping=False, patience=3):
    """Train the model with option to run full training without early stopping"""
    model.to(device)
    history = {'train_loss': [], 'train_acc': []}
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        start_time = time.time()
        
        try:
            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                
                if batch_count == 0:
                    print_gpu_memory()
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                
                optimizer.zero_grad()
                
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                
                loss.backward()
                optimizer.step()
                
               
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                
                torch.cuda.empty_cache()
                batch_count += 1
                
                
                if batch_count % 10 == 0:
                    batch_acc = 100 * correct / total
                    print(f"Batch {batch_count}, Loss: {loss.item():.4f}, Running Acc: {batch_acc:.2f}%")
            
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = 100 * correct / total
            epoch_time = time.time() - start_time
            
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, '
                  f'Time: {epoch_time:.1f}s')
            
            
            torch.save(model.state_dict(), f'densenet201_tb_epoch_{epoch+1}.pth')
            
           
            if use_early_stopping:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                    
                    torch.save(model.state_dict(), 'best_densenet201_tb.pth')
                    print("Saved best model checkpoint")
                else:
                    patience_counter += 1
                    print(f"No improvement in loss. Patience: {patience_counter}/{patience}")
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'best_densenet201_tb.pth')
                    print("Saved best model checkpoint")
                
        except Exception as e:
            print(f"Error during training: {e}")
            print(traceback.format_exc())
            
            torch.save(model.state_dict(), 'emergency_densenet201_tb.pth')
            print("Saved emergency checkpoint due to error")
            break
    
    
    try:
        model.load_state_dict(torch.load('best_densenet201_tb.pth'))
        print("Loaded best model for evaluation")
    except:
        print("Could not load best model, using current model")
    
    return model, history


def evaluate_model(model, test_loader, device):
    """Evaluate the model and calculate performance metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            
            torch.cuda.empty_cache()
        
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    
    if conf_matrix.size == 4:  
        tn, fp, fn, tp = conf_matrix.ravel()
    else:
        
        print("Warning: Could not unpack confusion matrix")
        tn, fp, fn, tp = 0, 0, 0, 0
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    
    try:
        precision = precision_score(all_labels, all_preds) * 100
    except:
        precision = 0
        
    try:
        recall = recall_score(all_labels, all_preds) * 100
    except:
        recall = 0
        
    try:
        f1 = f1_score(all_labels, all_preds) * 100
    except:
        f1 = 0
        
    try:
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    except:
        specificity = 0
    
    try:
        auc = roc_auc_score(all_labels, all_preds) * 100
    except:
        auc = 0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'auc': auc,
        'confusion_matrix': conf_matrix
    }
    
    return results

class ScoreCAM:
    """
    Implementation of Score-CAM visualization technique
    """
    def __init__(self, model, target_layer_name='features.denseblock4', device=None):
        self.model = model
        self.target_layer_name = target_layer_name
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.outputs = []
        self.handles = []
        
        
        for name, module in self.model.named_modules():
            if target_layer_name in name:
                self.handles.append(
                    module.register_forward_hook(self._get_activation)
                )
    
    def _get_activation(self, module, input, output):
        
        self.outputs.append(output)
    
    def generate_cam(self, input_image, target_class=None):
        self.outputs = []
        with torch.no_grad():
            output = self.model(input_image)
        if target_class is None:
            pred_score, pred_class = torch.max(output, dim=1)
            target_class = pred_class.item()
        else:
            pred_class = target_class
            pred_score = output[0, target_class].item()
        activation_maps = self.outputs[0]  
        batch_size, channels, height, width = activation_maps.shape
        cam = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        for i in range(channels):
            channel_map = activation_maps[0, i].unsqueeze(0).unsqueeze(0) 
            channel_map_norm = (channel_map - channel_map.min()) / (channel_map.max() - channel_map.min() + 1e-8)
            channel_map_upsampled = F.interpolate(
                channel_map_norm, 
                size=input_image.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            with torch.no_grad():
                masked_input = input_image * channel_map_upsampled
                output = self.model(masked_input)
                score = output[0, target_class]
            cam += score.item() * channel_map[0, 0]
        
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        
        cam = cam.cpu().numpy()
        
        return cam, pred_class, pred_score
    
    def visualize(self, image_path, target_class=None, use_segmented=False, segmentation_model=None):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if use_segmented and segmentation_model is not None:
            img = segment_lungs(img, segmentation_model)
        img_resized = cv2.resize(img, (224, 224))
        if img_resized.dtype != np.uint8:
            img_resized = np.clip(img_resized, 0, 1) * 255
            img_resized = img_resized.astype(np.uint8)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img_resized).unsqueeze(0).to(self.device)
        cam, pred_class, pred_score = self.generate_cam(img_tensor, target_class)
        cam_resized = cv2.resize(cam, (img_resized.shape[1], img_resized.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        cam_img = (heatmap * 0.4 + img_resized * 0.6).astype(np.uint8)
        return img_resized, cam_img, pred_class, pred_score
    
    def __del__(self):
        for handle in self.handles:
            handle.remove()


def visualize_score_cam(model, test_df, segmentation_model=None, device=None, num_samples=3):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normal_df = test_df[test_df['label'] == 'Normal'].sample(num_samples//2)
    tb_df = test_df[test_df['label'] == 'Tuberculosis'].sample(num_samples - num_samples//2)
    sample_df = pd.concat([normal_df, tb_df])
    score_cam = ScoreCAM(model, target_layer_name='features.denseblock4', device=device)
    fig, axes = plt.subplots(len(sample_df), 4, figsize=(16, 4*len(sample_df)))
    if len(sample_df) == 1:
        axes = np.expand_dims(axes, axis=0)
    # Process each image
    for i, (_, row) in enumerate(sample_df.iterrows()):
        try:
            img_path = row['image_path']
            label = row['label']
            if segmentation_model is not None and next(segmentation_model.parameters()).device != device:
                segmentation_model = segmentation_model.to(device)
            orig_img, orig_cam, orig_pred, orig_score = score_cam.visualize(
                img_path, 
                target_class=1 if label == 'Tuberculosis' else 0,
                use_segmented=False
            )
            seg_img, seg_cam, seg_pred, seg_score = score_cam.visualize(
                img_path, 
                target_class=1 if label == 'Tuberculosis' else 0,
                use_segmented=True,
                segmentation_model=segmentation_model
            )
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f"Original: {label}")
            axes[i, 0].axis('off')
            axes[i, 1].imshow(orig_cam)
            axes[i, 1].set_title(f"CAM on Original: {'TB' if orig_pred == 1 else 'Normal'} ({orig_score:.2f})")
            axes[i, 1].axis('off')
            axes[i, 2].imshow(seg_img)
            axes[i, 2].set_title(f"Segmented Lungs")
            axes[i, 2].axis('off')
            axes[i, 3].imshow(seg_cam)
            axes[i, 3].set_title(f"CAM on Segmented: {'TB' if seg_pred == 1 else 'Normal'} ({seg_score:.2f})")
            axes[i, 3].axis('off')
            
        except Exception as e:
            print(f"Error visualizing image {img_path}: {e}")
            traceback.print_exc()
    
    plt.tight_layout()
    plt.suptitle("Score-CAM Visualization: Where the model is focusing", fontsize=16, y=1.02)
    plt.savefig('score_cam_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    if segmentation_model is not None:
        segmentation_model = segmentation_model.cpu()
    torch.cuda.empty_cache()
    
    return fig

# Main function with optimized workflow and Score-CAM
def main():
    total_start_time = time.time()
    base_path = '/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database'
    normal_path = os.path.join(base_path, 'Normal')
    tb_path = os.path.join(base_path, 'Tuberculosis')
    
    def create_dataset_df(normal_dir, tb_dir):
        normal_files = glob(os.path.join(normal_dir, '*.png'))
        tb_files = glob(os.path.join(tb_dir, '*.png'))
        files = normal_files + tb_files
        labels = ['Normal'] * len(normal_files) + ['Tuberculosis'] * len(tb_files)
        return pd.DataFrame({
            'image_path': files,
            'label': labels
        })
    
    # Create dataframe
    print("Creating dataset dataframe...")
    df = create_dataset_df(normal_path, tb_path)
    
    print("\nDataset Information:")
    print(f"Total number of images: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    print("\nTrain/Test Split:")
    print(f"Training set: {len(train_df)} images")
    print(f"Testing set: {len(test_df)} images")
    
    try:
        
        segmentation_model = load_segmentation_model()
        
        
        cache_dir = '/kaggle/working/segmented_cache'
        train_df = preprocess_and_cache_segmented_images(train_df, segmentation_model, cache_dir)
        test_df = preprocess_and_cache_segmented_images(test_df, segmentation_model, cache_dir)
        
        
        segmentation_model = segmentation_model.cpu()
        torch.cuda.empty_cache()
        print_gpu_memory()
        
        
        print("\nCreating data loaders with pre-segmented lung images...")
        train_loader, test_loader = create_optimized_dataloaders(
            train_df, test_df, batch_size=8
        )
        model = setup_densenet201(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.001,
            momentum=0.9
        )
        print("\n===== Training DenseNet201 with Segmented Lungs =====")
        print_gpu_memory()
        model, history = train_model(
            model, train_loader, criterion, optimizer, device, 
            num_epochs=15,  
            use_early_stopping=False  
        )
        print("\n===== Evaluating Model =====")
        results = evaluate_model(model, test_loader, device)
        print("\n===== Results Summary =====")
        print(f"Accuracy: {results['accuracy']:.2f}% (Paper with segmentation: 99.90%)")
        print(f"Precision: {results['precision']:.2f}% (Paper with segmentation: 99.91%)")
        print(f"Sensitivity/Recall: {results['recall']:.2f}% (Paper with segmentation: 99.90%)")
        print(f"F1 Score: {results['f1_score']:.2f}% (Paper with segmentation: 99.90%)")
        print(f"Specificity: {results['specificity']:.2f}% (Paper with segmentation: 99.52%)")
        print(f"AUC: {results['auc']:.2f}%")
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'])
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
        plt.show()
        
        
        print("\n===== Generating Score-CAM Visualizations =====")
        try:
            
            segmentation_model = segmentation_model.to(device)
            
            
            fig = visualize_score_cam(
                model=model,
                test_df=test_df,
                segmentation_model=segmentation_model,
                device=device,
                num_samples=3
            )
            
            
            fig.savefig('score_cam_visualization.png', dpi=300, bbox_inches='tight')
            print("Score-CAM visualization saved")
            
            
            segmentation_model = segmentation_model.cpu()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error in Score-CAM visualization: {e}")
            traceback.print_exc()
        
        
        torch.save(model.state_dict(), 'densenet201_tb_segmented.pth')
        print("\nModel saved successfully.")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        print(traceback.format_exc())
    
    finally:
        
        total_time = time.time() - total_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
   
    try:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set
        pass
    
    main()


# %%

#initially results were comparable but then unet model which we used was changed a bit thus leading to large differences in values such as f1 values and sensitivity

# %% [markdown]
# # **Vision Transformers** 

# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, ViTForImageClassification, ViTConfig
import time
import gc
import traceback
from PIL import Image


def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print_gpu_memory()


def load_segmentation_model():
    print("Loading pretrained chest X-ray segmentation model...")
    try:
        model = AutoModel.from_pretrained("ianpan/chest-x-ray-basic", trust_remote_code=True)
        model = model.eval().to(device)
        print("Successfully loaded pretrained segmentation model")
        return model
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        print("Using fallback segmentation method")
        return None


def basic_lung_segmentation(image):
    """
    Basic lung segmentation using image processing as fallback
    """
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    
    _, binary = cv2.threshold(enhanced, 90, 255, cv2.THRESH_BINARY)
    
    
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    
    if len(image.shape) == 3:
        mask = np.stack([binary, binary, binary], axis=2) / 255.0
    else:
        mask = binary / 255.0
    
    segmented = image * mask
    
    return segmented


def segment_lungs(image, model=None):
    """
    Segment lungs from chest X-ray using the pretrained model
    """
    if model is None:
        return basic_lung_segmentation(image)
    orig_h, orig_w = image.shape[:2]
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = image
    
    try:
        x = model.preprocess(gray_img)  
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  
        x = x.float().to(device)    
        with torch.no_grad():
            out = model(x)
            torch.cuda.synchronize()  
        mask = out["mask"].squeeze().cpu().numpy()
        mask = mask.argmax(0) 
        binary_mask = np.zeros_like(mask)
        binary_mask[mask == 1] = 1 
        binary_mask[mask == 2] = 1  
        if binary_mask.shape[0] != orig_h or binary_mask.shape[1] != orig_w:
            binary_mask = cv2.resize(binary_mask.astype(np.float32), (orig_w, orig_h))
        if len(image.shape) == 3:
            segmented = image * np.expand_dims(binary_mask, axis=2)
        else:
            segmented = image * binary_mask
        if segmented.dtype == np.float32 or segmented.dtype == np.float64:
            segmented = np.clip(segmented, 0, 1)
            segmented = (segmented * 255).astype(np.uint8)
        return segmented
    
    except Exception as e:
        print(f"Error in segmentation: {e}")
        print(traceback.format_exc())
        return basic_lung_segmentation(image)

# Preprocess and cache segmented images
def preprocess_and_cache_segmented_images(df, model, cache_dir='segmented_cache'):
    os.makedirs(cache_dir, exist_ok=True)
    print("Pre-processing and caching segmented images...")
    processed_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        src_path = row['image_path']
        filename = os.path.basename(src_path)
        dest_path = os.path.join(cache_dir, filename)
        if os.path.exists(dest_path):
            processed_count += 1
            continue
        try:
            img = cv2.imread(src_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            segmented = segment_lungs(img, model)
            segmented = cv2.resize(segmented, (224, 224))
            cv2.imwrite(dest_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
            processed_count += 1
            if idx % 100 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing {src_path}: {e}")
    
    print(f"Processed {processed_count} images. Cached at {cache_dir}/")
    
    
    new_df = df.copy()
    new_df['segmented_path'] = new_df['image_path'].apply(
        lambda x: os.path.join(cache_dir, os.path.basename(x))
    )
    
    return new_df


class OptimizedChestXRayDataset(Dataset):
    def __init__(self, dataframe, transform=None, use_segmented_paths=True):
        self.dataframe = dataframe
        self.transform = transform
        self.use_segmented_paths = use_segmented_paths and 'segmented_path' in dataframe.columns
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        
        if self.use_segmented_paths:
            img_path = self.dataframe.iloc[idx]['segmented_path']
        else:
            img_path = self.dataframe.iloc[idx]['image_path']
            
        label = 1 if self.dataframe.iloc[idx]['label'] == 'Tuberculosis' else 0
        
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224))
        
        
        if self.transform:
            
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
            
            img = self.transform(img)
        
        return img, label


def get_vit_transforms(is_training=True):
    """Get image transformations optimized for ViT"""
    if is_training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
        ])


def create_optimized_dataloaders(train_df, test_df, batch_size=8):
    """Create data loaders with optimized settings for Kaggle GPU"""
    train_transform = get_vit_transforms(is_training=True)
    test_transform = get_vit_transforms(is_training=False)
    
    train_dataset = OptimizedChestXRayDataset(
        train_df, 
        transform=train_transform,
        use_segmented_paths=True
    )
    
    test_dataset = OptimizedChestXRayDataset(
        test_df, 
        transform=test_transform,
        use_segmented_paths=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  
        pin_memory=True  
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, test_loader


def setup_vit_model(num_classes=2, model_name="google/vit-base-patch16-224"):
    """Load and configure ViT for transfer learning"""
    print(f"Loading pretrained Vision Transformer: {model_name}...")
    
    
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    
    for param in model.vit.parameters():
        param.requires_grad = False
    
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    print(f"Vision Transformer loaded successfully with {num_classes} output classes")
    return model


def train_vit_model(model, train_loader, criterion, optimizer, device, num_epochs=15, use_early_stopping=False, patience=3):
    """Train the ViT model with optimized memory management"""
    model.to(device)
    history = {'train_loss': [], 'train_acc': []}
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        start_time = time.time()
        
        try:
            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                
                if batch_count == 0:
                    print_gpu_memory()
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                
                optimizer.zero_grad()
                
                
                outputs = model(inputs)
                loss = criterion(outputs.logits, labels)  
                
                
                loss.backward()
                optimizer.step()
                
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                
                torch.cuda.empty_cache()
                batch_count += 1
                
                
                if batch_count % 10 == 0:
                    batch_acc = 100 * correct / total
                    print(f"Batch {batch_count}, Loss: {loss.item():.4f}, Running Acc: {batch_acc:.2f}%")
            
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = 100 * correct / total
            epoch_time = time.time() - start_time
            
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, '
                  f'Time: {epoch_time:.1f}s')
            
            
            torch.save(model.state_dict(), f'vit_tb_epoch_{epoch+1}.pth')
            
            
            if use_early_stopping:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), 'best_vit_tb.pth')
                    print("Saved best model checkpoint")
                else:
                    patience_counter += 1
                    print(f"No improvement in loss. Patience: {patience_counter}/{patience}")
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'best_vit_tb.pth')
                    print("Saved best model checkpoint")
                
        except Exception as e:
            print(f"Error during training: {e}")
            print(traceback.format_exc())
            
            torch.save(model.state_dict(), 'emergency_vit_tb.pth')
            print("Saved emergency checkpoint due to error")
            break
    
    
    try:
        model.load_state_dict(torch.load('best_vit_tb.pth'))
        print("Loaded best model for evaluation")
    except:
        print("Could not load best model, using current model")
    
    return model, history
def evaluate_vit_model(model, test_loader, device):
    """Evaluate the ViT model with memory optimization"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.logits, 1)  
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy()) 
            torch.cuda.empty_cache()
   
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    
    if conf_matrix.size == 4:  
        tn, fp, fn, tp = conf_matrix.ravel()
    else:
       
        print("Warning: Could not unpack confusion matrix")
        tn, fp, fn, tp = 0, 0, 0, 0
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    
    try:
        precision = precision_score(all_labels, all_preds) * 100
    except:
        precision = 0
        
    try:
        recall = recall_score(all_labels, all_preds) * 100
    except:
        recall = 0
        
    try:
        f1 = f1_score(all_labels, all_preds) * 100
    except:
        f1 = 0
        
    try:
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    except:
        specificity = 0
    
    try:
        auc = roc_auc_score(all_labels, all_preds) * 100
    except:
        auc = 0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'auc': auc,
        'confusion_matrix': conf_matrix
    }
    
    return results


class ViTAttentionVisualizer:
    """Visualize attention maps from Vision Transformer models"""
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def visualize(self, image_path, use_segmented=False, segmentation_model=None):
        """Visualize attention maps for an image"""
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        if use_segmented and segmentation_model is not None:
            img = segment_lungs(img, segmentation_model)
        
        
        img_resized = cv2.resize(img, (224, 224))
        
        
        if img_resized.dtype != np.uint8:
            img_resized = np.clip(img_resized, 0, 1) * 255
            img_resized = img_resized.astype(np.uint8)
        
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img_tensor = transform(img_resized).unsqueeze(0).to(self.device)
        
        
        with torch.no_grad():
            outputs = self.model(img_tensor, output_attentions=True)
            logits = outputs.logits
            pred_class = torch.argmax(logits, dim=-1).item()
            pred_score = torch.softmax(logits, dim=-1)[0, pred_class].item()
            
            
            attn = outputs.attentions[-1]  
            
            
            attn = attn.mean(dim=1)[0]  
            
            
            cls_attn = attn[0, 1:]  
            
            
            patch_size = 16  
            grid_size = 224 // patch_size  
            attn_map = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
            
            
            attn_map = cv2.resize(attn_map, (img_resized.shape[1], img_resized.shape[0]))
            
            
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            
            attn_img = (heatmap * 0.4 + img_resized * 0.6).astype(np.uint8)
            
            return img_resized, attn_img, pred_class, pred_score


def visualize_attention(model, test_df, segmentation_model=None, device=None, num_samples=3):
    """Visualize attention for a few sample images"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    normal_df = test_df[test_df['label'] == 'Normal'].sample(num_samples//2)
    tb_df = test_df[test_df['label'] == 'Tuberculosis'].sample(num_samples - num_samples//2)
    sample_df = pd.concat([normal_df, tb_df])
    
    
    visualizer = ViTAttentionVisualizer(model, device=device)
    
    
    fig, axes = plt.subplots(len(sample_df), 4, figsize=(16, 4*len(sample_df)))
    
    
    if len(sample_df) == 1:
        axes = np.expand_dims(axes, axis=0)
    
    
    for i, (_, row) in enumerate(sample_df.iterrows()):
        try:
            
            img_path = row['image_path']
            label = row['label']
            
            
            if segmentation_model is not None and next(segmentation_model.parameters()).device != device:
                segmentation_model = segmentation_model.to(device)
            
            
            orig_img, orig_attn, orig_pred, orig_score = visualizer.visualize(
                img_path, 
                use_segmented=False
            )
            
            
            seg_img, seg_attn, seg_pred, seg_score = visualizer.visualize(
                img_path, 
                use_segmented=True,
                segmentation_model=segmentation_model
            )
            
            # Plot results
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f"Original: {label}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(orig_attn)
            axes[i, 1].set_title(f"ViT Attention: {'TB' if orig_pred == 1 else 'Normal'} ({orig_score:.2f})")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(seg_img)
            axes[i, 2].set_title(f"Segmented Lungs")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(seg_attn)
            axes[i, 3].set_title(f"ViT on Segmented: {'TB' if seg_pred == 1 else 'Normal'} ({seg_score:.2f})")
            axes[i, 3].axis('off')
            
        except Exception as e:
            print(f"Error visualizing image {img_path}: {e}")
            traceback.print_exc()
    
    plt.tight_layout()
    plt.suptitle("ViT Attention Visualization: Where the model is focusing", fontsize=16, y=1.02)
    plt.savefig('vit_attention_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    if segmentation_model is not None:
        segmentation_model = segmentation_model.cpu()
    torch.cuda.empty_cache()
    
    return fig


def main():
    
    total_start_time = time.time()
    
    
    base_path = '/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database'
    normal_path = os.path.join(base_path, 'Normal')
    tb_path = os.path.join(base_path, 'Tuberculosis')
    
    def create_dataset_df(normal_dir, tb_dir):
        normal_files = glob(os.path.join(normal_dir, '*.png'))
        tb_files = glob(os.path.join(tb_dir, '*.png'))
        
        files = normal_files + tb_files
        labels = ['Normal'] * len(normal_files) + ['Tuberculosis'] * len(tb_files)
        
        return pd.DataFrame({
            'image_path': files,
            'label': labels
        })
    
    
    print("Creating dataset dataframe...")
    df = create_dataset_df(normal_path, tb_path)
    
    print("\nDataset Information:")
    print(f"Total number of images: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    print("\nTrain/Test Split:")
    print(f"Training set: {len(train_df)} images")
    print(f"Testing set: {len(test_df)} images")
    
    try:
        
        segmentation_model = load_segmentation_model()
        
        
        cache_dir = '/kaggle/working/segmented_cache'
        train_df = preprocess_and_cache_segmented_images(train_df, segmentation_model, cache_dir)
        test_df = preprocess_and_cache_segmented_images(test_df, segmentation_model, cache_dir)
        
        
        segmentation_model = segmentation_model.cpu()
        torch.cuda.empty_cache()
        print_gpu_memory()
        
        
        print("\nCreating data loaders with pre-segmented lung images...")
        train_loader, test_loader = create_optimized_dataloaders(
            train_df, test_df, batch_size=8
        )
        
        
        model = setup_vit_model(num_classes=2, model_name="google/vit-base-patch16-224")
        model = model.to(device)
        
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=0.01
        )
        
        
        print("\n===== Training Vision Transformer with Segmented Lungs =====")
        print_gpu_memory()
        model, history = train_vit_model(
            model, train_loader, criterion, optimizer, device, 
            num_epochs=15,
            use_early_stopping=False
        )
        
        
        print("\n===== Evaluating ViT Model =====")
        results = evaluate_vit_model(model, test_loader, device)
        
        
        print("\n===== Results Summary =====")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print(f"Precision: {results['precision']:.2f}%")
        print(f"Sensitivity/Recall: {results['recall']:.2f}%")
        print(f"F1 Score: {results['f1_score']:.2f}%")
        print(f"Specificity: {results['specificity']:.2f}%")
        print(f"AUC: {results['auc']:.2f}%")
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
        
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'])
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig('vit_training_history.png', dpi=300)
        plt.show()
        
        
        print("\n===== Generating ViT Attention Visualizations =====")
        try:
            
            visualize_attention(
                model=model,
                test_df=test_df,
                segmentation_model=segmentation_model,
                device=device,
                num_samples=3
            )
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            traceback.print_exc()
        
        
        torch.save(model.state_dict(), 'vit_tb_segmented.pth')
        print("\nModel saved successfully.")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        print(traceback.format_exc())
    
    finally:
        
        total_time = time.time() - total_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
    try:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        
        pass
    
    main()


# %% [markdown]
# ## Effecient VIT

# %%


import os, cv2, torch, numpy as np, pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = '/kaggle/working/segmented_cache'


def basic_lung_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 90, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    mask = binary / 255.0
    return (image * mask[:, :, None]).astype(np.uint8) if image.ndim == 3 else image * mask

def preprocess_and_cache(df):
    os.makedirs(CACHE_DIR, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching segmentation"):
        src = row['image_path']
        dst = os.path.join(CACHE_DIR, os.path.basename(src))
        if not os.path.exists(dst):
            img = cv2.imread(src)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            seg = basic_lung_segmentation(img)
            seg = cv2.resize(seg, (224, 224))
            cv2.imwrite(dst, cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))
    df['segmented_path'] = df['image_path'].apply(lambda p: os.path.join(CACHE_DIR, os.path.basename(p)))
    return df


def get_transforms(train):
    ts = [transforms.ToPILImage(), transforms.Resize((224, 224))]
    if train:
        ts += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    ts += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    return transforms.Compose(ts)

class TBChestXRayDataset(Dataset):
    def __init__(self, df, transform): self.df, self.transform = df.reset_index(drop=True), transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        img = cv2.imread(self.df.loc[i, 'segmented_path'], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.transform: img = self.transform(img)
        label = 1 if self.df.loc[i, 'label'] == 'Tuberculosis' else 0
        return img, label


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, groups=1, bn_init=1):
        conv = nn.Conv2d(a, b, ks, stride, pad, groups=groups, bias=False)
        bn = nn.BatchNorm2d(b)
        super().__init__(conv, bn)
        nn.init.constant_(bn.weight, bn_init)
        nn.init.constant_(bn.bias, 0)

class Residual(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return x + self.m(x)

class FFN(nn.Module):
    def __init__(self, ed, h): super().__init__(); self.ffn = nn.Sequential(Conv2d_BN(ed, h), nn.ReLU(), Conv2d_BN(h, ed, bn_init=0))
    def forward(self, x): return self.ffn(x)

class MedicalXRayAttention(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.roi = nn.Sequential(nn.Conv2d(C, C//2, 3, 1, 1), nn.ReLU(), nn.Conv2d(C//2, 4, 1), nn.Sigmoid())
        self.ca  = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(C, C//16, 1), nn.ReLU(), nn.Conv2d(C//16, C, 1), nn.Sigmoid())
        self.sa  = nn.Sequential(nn.Conv2d(2, 1, 7, 1, 3), nn.Sigmoid())
    def forward(self, x):
        B, C, H, W = x.shape
        coords = self.roi(x).mean((2, 3))
        out = []
        for i in range(B):
            x1, y1, x2, y2 = (coords[i] * torch.tensor([W, H, W, H], device=x.device)).int()
            x1, y1 = x1.clamp(0, W-1), y1.clamp(0, H-1)
            x2, y2 = x2.clamp(x1+1, W), y2.clamp(y1+1, H)
            crop = x[i:i+1, :, y1:y2, x1:x2]
            out.append(F.interpolate(crop, (H, W), mode='bilinear', align_corners=False))
        x = torch.cat(out, 0)
        x = x * self.ca(x)
        pooled = torch.cat([x.max(1, keepdim=True)[0], x.mean(1, keepdim=True)], dim=1)
        return x * self.sa(pooled)

class EfficientViTBlock(nn.Module):
    def __init__(self, ed):
        super().__init__()
        self.ffn1 = Residual(FFN(ed, ed * 2))
        self.mxa  = Residual(MedicalXRayAttention(ed))
        self.ffn2 = Residual(FFN(ed, ed * 2))
    def forward(self, x):
        return self.ffn2(self.mxa(self.ffn1(x)))

class EfficientViT(nn.Module):
    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()
        self.patch  = nn.Sequential(Conv2d_BN(in_ch, 64, 7, 2, 3), nn.ReLU(), Conv2d_BN(64, 128, 3, 2, 1), nn.ReLU())
        self.blocks = nn.Sequential(*[EfficientViTBlock(128) for _ in range(6)])
        self.head   = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.patch(x)
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x).squeeze(1)

def train_epoch(model, dl, crit, opt):
    model.train(); total, correct, loss_sum = 0, 0, 0.0
    for x, y in dl:
        x, y = x.to(device), y.float().to(device)
        opt.zero_grad(); logits = model(x); loss = crit(logits, y)
        loss.backward(); opt.step()
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == y.long()).sum().item(); total += x.size(0)
        loss_sum += loss.item() * x.size(0)
    return loss_sum / total, correct / total

def eval_model(model, dl):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x.to(device))
            preds += (torch.sigmoid(logits) > 0.5).long().cpu().tolist()
            labels += y.tolist()
    return metrics.accuracy_score(labels, preds), metrics.roc_auc_score(labels, preds)


class GradCAMVisualizer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        def save_grad(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        def save_activation(module, input, output):
            self.activations = output
        self.model.blocks[-1].register_forward_hook(save_activation)
        self.model.blocks[-1].register_full_backward_hook(save_grad)

    def generate(self, image_tensor, raw_image, save_path):
        image_tensor = image_tensor.unsqueeze(0).to(device)
        self.model.zero_grad()
        output = self.model(image_tensor)
        pred = torch.sigmoid(output).item()
        output.backward()
        grads = self.gradients.squeeze().mean(dim=(1, 2))
        act = self.activations.squeeze()
        cam = torch.sum(grads[:, None, None] * act, dim=0).cpu().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = (heatmap * 0.4 + raw_image * 0.6).astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return pred

def visualize_gradcam(model, sample_df, transform):
    visualizer = GradCAMVisualizer(model)
    for _, row in sample_df.iterrows():
        img_path = row['segmented_path']
        raw_img = cv2.imread(img_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        tensor_img = transform(raw_img)
        pred = visualizer.generate(tensor_img, raw_img, f"/kaggle/working/cam_{os.path.basename(img_path)}")


def main():
    base = '/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database'
    df = pd.DataFrame({
        'image_path': glob(f"{base}/Normal/*.png") + glob(f"{base}/Tuberculosis/*.png"),
        'label': ['Normal'] * len(glob(f"{base}/Normal/*.png")) + ['Tuberculosis'] * len(glob(f"{base}/Tuberculosis/*.png"))
    })

    df = preprocess_and_cache(df)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_dl = DataLoader(TBChestXRayDataset(train_df, get_transforms(True)), batch_size=16, shuffle=True, num_workers=2)
    test_dl  = DataLoader(TBChestXRayDataset(test_df,  get_transforms(False)), batch_size=16, shuffle=False, num_workers=2)

    model = EfficientViT().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(15):
        loss, acc = train_epoch(model, train_dl, criterion, optimizer)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")

    acc, auc = eval_model(model, test_dl)
    print(f"Final Test Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    torch.save(model.state_dict(), '/kaggle/working/efficientvit_tb.pth')

    visualize_gradcam(model, test_df.sample(3), transform=get_transforms(False))

if __name__ == '__main__':
    main()


# %% [markdown]
# ## Visualization

# %%

import matplotlib.pyplot as plt
import cv2
import glob

def show_cam_images(cam_dir="/kaggle/working", num_images=3):
    cam_paths = sorted(glob.glob(f"{cam_dir}/cam_*.png"))[:num_images]
    if not cam_paths:
        print("No CAM images found.")
        return

    plt.figure(figsize=(5 * num_images, 5))
    for i, path in enumerate(cam_paths):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(os.path.basename(path), fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# showing top 3 images
show_cam_images()



