import os
import streamlit as st
import requests
from tqdm import tqdm
import time
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_html_content(content):
    """Check if content appears to be HTML"""
    content_str = content[:1024].decode('utf-8', errors='ignore').lower()
    return ('<!doctype html' in content_str or 
            '<html' in content_str or 
            'google drive' in content_str or 
            'sign in' in content_str or
            'too many users' in content_str or
            'download warning' in content_str)

def get_confirm_token(content):
    """Extract confirmation token from HTML content"""
    content_str = content.decode('utf-8', errors='ignore')
    match = re.search(r'confirm=([0-9A-Za-z_]+)', content_str)
    if match:
        return match.group(1)
    
    # Alternative pattern for newer Google Drive pages
    match = re.search(r'name="confirm" value="([^"]+)"', content_str)
    if match:
        return match.group(1)
    
    return None

def download_file_from_google_drive(file_id, destination, max_retries=3):
    """Download a file from Google Drive with robust error handling"""
    URL = "https://docs.google.com/uc?export=download"
    
    for attempt in range(max_retries):
        try:
            session = requests.Session()
            
            # Initial request
            response = session.get(URL, params={'id': file_id}, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check for confirmation token in cookies
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    params = {'id': file_id, 'confirm': value}
                    response = session.get(URL, params=params, stream=True, timeout=30)
                    break
            
            # If we got HTML content, try to extract confirmation token
            if is_html_content(response.content):
                token = get_confirm_token(response.content)
                if token:
                    st.info(f"Found confirmation token, attempt {attempt+1}/{max_retries}")
                    params = {'id': file_id, 'confirm': token}
                    response = session.get(URL, params=params, stream=True, timeout=30)
            
            # If still getting HTML, check for "Too many users" message
            if is_html_content(response.content):
                content_str = response.content[:1024].decode('utf-8', errors='ignore').lower()
                if 'too many users' in content_str:
                    wait_time = 2 ** attempt  # Exponential backoff
                    st.warning(f"Google Drive rate limit hit. Waiting {wait_time} seconds before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
            
            # Check if we finally got binary content
            if is_html_content(response.content):
                # Save the HTML for debugging
                with open("debug_google_drive.html", "wb") as f:
                    f.write(response.content)
                raise ValueError("Failed to download file - received HTML content after multiple attempts")
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(destination, "wb") as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
                    for chunk in response.iter_content(32768):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Verify download
            if os.path.getsize(destination) < 10000:  # Less than 10KB is suspicious
                raise ValueError(f"Downloaded file is too small ({os.path.getsize(destination)} bytes)")
                
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                st.warning(f"Download attempt {attempt+1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error(f"Failed to download after {max_retries} attempts: {str(e)}")
                raise

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
        if not download_file_from_google_drive(resnet_file_id, resnet_path):
            raise Exception("ResNet model download failed")
        
        # Download SVM model
        st.info(f"Downloading SVM model...")
        if not download_file_from_google_drive(svm_file_id, svm_path):
            raise Exception("SVM model download failed")
        
        st.success("Models downloaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error downloading models: {str(e)}")
        st.error("Please contact the administrator for assistance.")
        return False
