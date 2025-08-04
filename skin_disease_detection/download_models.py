import os
import streamlit as st

def download_models():
    """Models are already in the repository thanks to Git LFS"""
    models_dir = "models"
    
    # Check if models already exist
    resnet_path = os.path.join(models_dir, "resnet50_base_model.h5")
    svm_path = os.path.join(models_dir, "svm_model_optimized.pkl")
    
    if os.path.exists(resnet_path) and os.path.exists(svm_path):
        st.success("Models are ready to use!")
        return True
    
    st.error("Error: Model files not found in repository. Please contact administrator.")
    return False
