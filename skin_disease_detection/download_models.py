import os
import gdown
import streamlit as st
import re
from urllib.parse import urlparse

def clean_google_drive_url(url):
    """Ensure Google Drive URL is in the correct format for gdown"""
    # Extract file ID from various possible URL formats
    parsed = urlparse(url)
    
    # Case 1: Standard sharing link: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    m = re.search(r"/file/d/([-\w_]+)/", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?id={file_id}"
    
    # Case 2: Direct ID format: https://drive.google.com/open?id=FILE_ID
    m = re.search(r"id=([-\w_]+)", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?id={file_id}"
    
    # Case 3: Already in uc format
    if "drive.google.com/uc" in url and "id=" in url:
        return url
    
    # If we can't parse it, return original (gdown might handle it with fuzzy=True)
    return url

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
    
    # Original URLs (could be in any format)
    resnet_url = "https://drive.google.com/file/d/182OmtnTmFW8WfHEw1tHwnUhMp4VkZFGC/view?usp=sharing"
    svm_url = "https://drive.google.com/file/d/1kWnR-WP70b-JvCbmpc81wdrx0bhOYUo-/view?usp=sharing"
    
    # Clean and standardize the URLs
    resnet_url = clean_google_drive_url(resnet_url)
    svm_url = clean_google_drive_url(svm_url)
    
    st.info(f"Using cleaned ResNet URL: {resnet_url}")
    st.info(f"Using cleaned SVM URL: {svm_url}")
    
    try:
        # Download ResNet model
        gdown.download(resnet_url, resnet_path, quiet=False, fuzzy=True)
        
        # Download SVM model
        gdown.download(svm_url, svm_path, quiet=False, fuzzy=True)
        
        # Verify the files are actual model files, not HTML error pages
        if os.path.getsize(resnet_path) < 100000:  # HDF5 files should be large
            raise ValueError("ResNet model file is too small (likely an HTML error page)")
        if os.path.getsize(svm_path) < 100000:  # Pickle files should be large
            raise ValueError("SVM model file is too small (likely an HTML error page)")
        
        st.success("Models downloaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error downloading models: {str(e)}")
        st.error("Please contact the administrator for assistance.")
        return False
