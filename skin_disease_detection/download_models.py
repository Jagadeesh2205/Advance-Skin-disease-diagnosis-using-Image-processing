import os
import streamlit as st
import requests
from tqdm import tqdm
import re

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive with proper confirmation handling"""
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    
    # Initial request
    response = session.get(URL, params={'id': file_id}, stream=True)
    response.raise_for_status()
    
    # Check if we need to confirm the download
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
            break
    else:
        # Check for confirmation token in the HTML content
        content = response.content
        match = re.search(r'confirm=([0-9A-Za-z_]+)', content.decode('utf-8'))
        if match:
            params = {'id': file_id, 'confirm': match.group(1)}
            response = session.get(URL, params=params, stream=True)
    
    # Get file size if available
    total_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    with open(destination, "wb") as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    # Verify we got a valid file (not an HTML error page)
    with open(destination, 'rb') as f:
        content = f.read(1024)  # Read first KB to check content
        if b'<!DOCTYPE html>' in content or b'<html' in content:
            raise ValueError("Downloaded file appears to be an HTML page (likely a Google Drive error)")

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
    
    # File IDs only (no URLs needed)
    resnet_file_id = "182OmtnTmFW8WfHEw1tHwnUhMp4VkZFGC"
    svm_file_id = "1kWnR-WP70b-JvCbmpc81wdrx0bhOYUo-"
    
    try:
        # Download ResNet model
        st.info(f"Downloading ResNet model...")
        download_file_from_google_drive(resnet_file_id, resnet_path)
        
        # Download SVM model
        st.info(f"Downloading SVM model...")
        download_file_from_google_drive(svm_file_id, svm_path)
        
        # Verify the files are actual model files
        if os.path.getsize(resnet_path) < 100000:  # Should be several MB
            raise ValueError("ResNet model file is too small (likely an HTML error page)")
        if os.path.getsize(svm_path) < 10000:  # SVM models are smaller but still >10KB
            raise ValueError("SVM model file is too small (likely an HTML error page)")
        
        st.success("Models downloaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error downloading models: {str(e)}")
        st.error("Please contact the administrator for assistance.")
        return False
