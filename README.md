# 🏥 Skin Disease Detection System using Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10-red.svg)](https://www.tensorflow.org/)

[Skin Disease Detection Demo]((https://jagadeesh22-skin-disease-detection.hf.space/?logs=container&__theme=system&deep_link=YlyTvJfTuhQ))

## 📌 Project Overview

This project implements a deep learning-based skin disease detection system that can identify 7 different skin conditions with high accuracy. The system uses a hybrid approach combining deep feature extraction with ResNet50 and classification with SVM to achieve robust performance even on limited hardware resources.

### ✨ Key Features:

* **7-class skin disease classification** (Chickenpox, Cellulitis, Athlete's Foot, Impetigo, Nail Fungus, Ringworm, Cutaneous Larva Migrans)
* **Detailed medical reports** with causes, symptoms, treatments, and prevention strategies
* **Responsive UI** with image upload and analysis capabilities
* **Optimized for laptop GPUs** with limited VRAM (4GB)
* **Educational purpose** - provides valuable information about skin conditions

## ⚠️ Important Disclaimer

**THIS APPLICATION IS FOR EDUCATIONAL PURPOSES ONLY AND IS NOT A MEDICAL DIAGNOSTIC TOOL.** The results provided by this system should never replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns.

## 🌟 Project Significance

This project demonstrates how deep learning can be made accessible even on consumer-grade hardware through careful optimization. It shows that with the right techniques, meaningful medical image analysis can be performed on standard laptops rather than requiring expensive server infrastructure.

The system is particularly valuable for:

* Medical students learning about skin conditions
* Patients seeking preliminary information before consulting a doctor
* Remote areas with limited access to dermatologists
* Educational institutions teaching medical imaging applications

## 💻 Hardware Requirements

| Component   | Minimum Requirement                        | Recommended                         |
| ----------- | ------------------------------------------ | ----------------------------------- |
| **GPU**     | NVIDIA GPU with 4GB+ VRAM (e.g., GTX 1650) | NVIDIA RTX 3050/3060 with 6GB+ VRAM |
| **CPU**     | Quad-core processor                        | 8+ core processor                   |
| **RAM**     | 8GB                                        | 16GB+                               |
| **Storage** | 5GB free space                             | 10GB+ free space                    |

**Note for Laptop Users:** This project has been specifically optimized to run on laptop GPUs with 4GB VRAM (like NVIDIA RTX 3050 Laptop GPU) through various memory optimization techniques.

## 🛠️ Software Requirements

### Core Dependencies:

* **Python 3.9** (required for TensorFlow 2.10 compatibility)
* **TensorFlow 2.10.0** (with GPU support)
* **CUDA Toolkit 11.2**
* **cuDNN 8.1.0** (for CUDA 11.2)
* **Conda** (for environment management)

### Python Libraries:

```requirements.txt
Flask==2.2.5
numpy==1.21.6
opencv-python==4.5.5.64
scikit-learn==1.0.2
tensorflow==2.10.0
pillow==9.0.0
gunicorn==20.1.0
```

## ⚙️ Installation Process

### Step 1: Set up Conda Environment

```bash
# Create a new conda environment
conda create -n skin-disease python=3.9
conda activate skin-disease

# Install core dependencies
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# Install Python packages
pip install -r requirements.txt
```

### Step 2: Verify GPU Setup

Create a test file `gpu_test.py`:

```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("TensorFlow Version:", tf.__version__)
```

Run it to verify GPU detection:

```bash
python gpu_test.py
```

You should see output similar to:

```
GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
TensorFlow Version: 2.10.0
```

### Step 3: Configure Environment Variables

Add these to the top of your Python scripts (before importing TensorFlow):

```python
import os
# Memory optimization flags
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = ''
os.environ['XLA_FLAGS'] = ''

# Enable mixed precision for memory savings
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

### Step 4: Configure GPU Memory

For laptop GPUs with 4GB VRAM, add this after GPU setup:

```python
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        # Set memory limit to 2.5GB (leaves 1.5GB headroom for 4GB GPU)
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2560)]
        )
```


## 🚀 Running the Application

1. **Activate your conda environment**:

   ```bash
   conda activate skin-disease
   ```

2. **Start the Flask application**:

   ```bash
   python app.py
   ```

3. **Access the application** in your browser:

   ```
   http://localhost:5000
   ```

## 🧪 Troubleshooting Common Issues

### 1. GPU Memory Exhaustion

**Symptoms**: `RESOURCE_EXHAUSTED` errors during feature extraction
**Solution**:

* Reduce batch size to 1 (already implemented in this project)
* Use mixed precision (`mixed_float16`)
* Set GPU memory limit (as shown in Step 4 above)
* Reduce image size to 192x192 (already implemented)

### 2. libdevice Not Found Error

**Symptoms**: `libdevice not found at ./libdevice.10.bc`
**Solution**:

* Disable XLA JIT compilation by setting:

  ```python
  os.environ['TF_XLA_FLAGS'] = ''
  os.environ['XLA_FLAGS'] = ''
  ```

### 3. Model Loading Issues

**Symptoms**: Errors loading `.pkl` or `.h5` files
**Solution**:

* Ensure models are in the correct directory
* Verify scikit-learn version matches training environment
* Use direct pickle loading instead of manual reconstruction

## 💡 Key Lessons & Recommendations

### For Developers Working on Similar Projects:

1. **Memory Management is Critical**:

   * Laptop GPUs (4GB VRAM) require special handling
   * Always implement chunked processing for large datasets
   * Mixed precision (`float16`) provides \~40% memory savings with minimal accuracy impact

2. **Optimization Techniques That Worked**:

   * Reducing image size from 224x224 → 192x192 (30% memory reduction)
   * Using batch size 1 for feature extraction on limited VRAM
   * Implementing memory cleanup between processing chunks
   * Disabling XLA JIT compilation to avoid libdevice errors

3. **Hardware Considerations**:

   * NVIDIA GPUs work better with TensorFlow than AMD
   * Verify CUDA/cuDNN version compatibility with TensorFlow
   * Leave 1GB+ VRAM headroom for system operations

4. **For Medical Applications**:

   * Always include prominent disclaimers about educational use only
   * Provide references to professional medical resources
   * Consider privacy implications of user-uploaded medical images


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

* TensorFlow and Keras teams for open-source deep learning frameworks
* scikit-learn developers for robust machine learning tools
* Medical professionals who contributed disease information
* The open-source community for valuable resources and support

---

*This project was developed with educational purposes in mind. Always consult healthcare professionals for medical concerns.*
