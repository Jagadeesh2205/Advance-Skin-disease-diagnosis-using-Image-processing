import os
import streamlit as st
import logging

# Set up logging
logger = logging.getLogger(__name__)

def download_models():
    """Models are already in the repository - verify exact location"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Script directory: {script_dir}")
    
    # The models should be in the models directory relative to this script
    models_dir = os.path.join(script_dir, "models")
    logger.info(f"Looking for models in: {models_dir}")
    
    # Check if all required model files exist
    svm_path = os.path.join(models_dir, "svm_model_optimized.pkl")
    svm_sv_path = os.path.join(models_dir, "svm_model_optimized_support_vectors.npy")
    resnet_path = os.path.join(models_dir, "resnet50_base_model.h5")
    
    svm_exists = os.path.exists(svm_path)
    svm_sv_exists = os.path.exists(svm_sv_path)
    resnet_exists = os.path.exists(resnet_path)
    
    logger.info(f"SVM model exists: {svm_exists}")
    logger.info(f"SVM support vectors exist: {svm_sv_exists}")
    logger.info(f"ResNet model exists: {resnet_exists}")
    
    if svm_exists and svm_sv_exists and resnet_exists:
        st.success("Models are ready to use!")
        return True
    
    # Show detailed error about which files are missing
    missing_files = []
    if not svm_exists:
        missing_files.append("svm_model_optimized.pkl")
    if not svm_sv_exists:
        missing_files.append("svm_model_optimized_support_vectors.npy")
    if not resnet_exists:
        missing_files.append("resnet50_base_model.h5")
    
    st.error(f"Error: Model files not found in repository. Missing: {', '.join(missing_files)}")
    st.error(f"Script directory: {script_dir}")
    st.error(f"Looking in: {models_dir}")
    return False
