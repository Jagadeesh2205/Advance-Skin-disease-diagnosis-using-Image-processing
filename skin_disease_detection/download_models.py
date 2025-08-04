import os
import gdown
import zipfile
import streamlit as st

def download_models():
    """Download models from Google Drive if they don't exist locally"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Check if models already exist
    resnet_path = os.path.join(models_dir, "resnet50_base_model.h5")
    svm_path = os.path.join(models_dir, "svm_model_optimized.pkl")
    
    if os.path.exists(resnet_path) and os.path.exists(svm_path):
        return True
    
    st.info("Downloading model files (this may take a few minutes)...")
    
    # FIX: Use direct download URLs (uc?id= format)
    resnet_url = "https://drive.google.com/uc?id=182OmtnTmFW8WfHEw1tHwnUhMp4VkZFGC"
    svm_url = "https://drive.google.com/uc?id=1kWnR-WP70b-JvCbmpc81wdrx0bhOYUo-"
    
    try:
        # Download ResNet model
        gdown.download(resnet_url, resnet_path, quiet=False, fuzzy=True)
        
        # Download SVM model
        gdown.download(svm_url, svm_path, quiet=False, fuzzy=True)
        
        st.success("Models downloaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error downloading models: {str(e)}")
        st.error("Please contact the administrator for assistance.")
        return False
