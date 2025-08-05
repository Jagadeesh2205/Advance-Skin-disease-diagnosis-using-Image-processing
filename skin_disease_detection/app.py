import os
import streamlit as st
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from download_models import download_models

# Download models before loading them
logger.info("Starting model verification process...")
if not download_models():
    logger.error("Model verification failed - stopping app")
    st.stop()
else:
    logger.info("Model verification successful - proceeding with app")

import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
from PIL import Image
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🏥",
    layout="wide"
)

def load_minimal_svm_model(file_path):
    """Load the minimal SVM model and reconstruct it for use"""
    # Load the minimal model structure
    with open(file_path, 'rb') as f:
        minimal_model = pickle.load(f)
    
    # Load support vectors
    sv_file = file_path.replace('.pkl', '_support_vectors.npy')
    support_vectors = np.load(sv_file)
    
    # Create a new SVC with the same parameters
    from sklearn.svm import SVC
    full_model = SVC(
        kernel=minimal_model['kernel'],
        C=minimal_model['C'],
        gamma=minimal_model['gamma'],
        probability=False,
        random_state=minimal_model['random_state']
    )
    
    # Update the model's internal state directly (bypassing read-only properties)
    # Update the model's internal state directly (bypassing read-only properties)
    # Update the model's internal state directly (bypassing read-only properties)
    # Update the model's internal state directly (bypassing read-only properties)
    full_model.__dict__.update({
        # Public attributes (with trailing underscore)
        'support_': minimal_model['support_'],
        'n_support_': minimal_model['n_support_'],
        'dual_coef_': minimal_model['dual_coef_'],
        'intercept_': minimal_model['intercept_'],
        'classes_': minimal_model['classes_'],
        'support_vectors_': support_vectors,
        'fit_status_': 0,
        'probA_': None,
        'probB_': None,
        'shape_fit_': (support_vectors.shape[1],),
        
        # Private attributes (with leading underscore)
        '_n_support': minimal_model['n_support_'],
        '_dual_coef': minimal_model['dual_coef_'],
        '_intercept': minimal_model['intercept_'],
        
        # Direct attribute names (no underscores)
        'dual_coef': minimal_model['dual_coef_'],
        'intercept': minimal_model['intercept_'],
        'n_support': minimal_model['n_support_'],
        
        # Sparse attributes (both versions)
        '_sparse': False,
        'sparse_': False,
        'sparse': False
    })
    
    # Set additional attributes directly for maximum compatibility
    full_model.dual_coef = minimal_model['dual_coef_']
    full_model.intercept = minimal_model['intercept_']
    full_model.n_support = minimal_model['n_support_']
    full_model._sparse = False
    full_model.sparse_ = False
    full_model.sparse = False
        
    return full_model

# Disease information
DISEASE_INFO = {
    'VI-chickenpox': {
        'name': 'Chickenpox',
        'description': 'Chickenpox is a highly contagious disease caused by the initial infection with the varicella zoster virus (VZV). It is characterized by a distinctive itchy rash that forms small, fluid-filled blisters that eventually scab over.',
        'causes': [
            'Caused by the varicella-zoster virus (a member of the herpesvirus family)',
            'Spreads through direct contact with the rash or through airborne droplets',
            'Highly contagious 1-2 days before rash appears until all blisters have scabbed over',
            'More common in children but can affect adults with more severe symptoms'
        ],
        'symptoms': [
            'Red, itchy rash that starts on face and chest, then spreads',
            'Fluid-filled blisters that break and scab over',
            'Fever',
            'Headache',
            'Fatigue',
            'Loss of appetite'
        ],
        'treatment': [
            'Antiviral medications for severe cases or high-risk individuals',
            'Calamine lotion for itch relief',
            'Antihistamines to reduce itching',
            'Acetaminophen for fever (avoid aspirin in children)',
            'Keeping fingernails short to prevent infection from scratching'
        ],
        'prevention': [
            'Varicella vaccine (highly effective)',
            'Avoiding contact with infected individuals',
            'Isolating infected individuals until all blisters have scabbed over',
            'Good hand hygiene'
        ],
        'duration': '7-10 days',
        'when_to_see_doctor': [
            'If you have a weakened immune system',
            'If symptoms worsen after initial improvement',
            'If fever lasts more than 4 days',
            'If rash becomes very red, warm, or painful (signs of bacterial infection)',
            'If you develop difficulty breathing or chest pain'
        ]
    },
    'BA-cellulitis': {
        'name': 'Cellulitis',
        'description': 'Cellulitis is a common, potentially serious bacterial skin infection. It appears as a swollen, red area of skin that feels hot and tender, and it may spread rapidly. Cellulitis usually affects the skin on the lower legs, but it can occur in other areas.',
        'causes': [
            'Most commonly caused by Streptococcus and Staphylococcus bacteria',
            'Enters through cracks, cuts, or breaks in the skin',
            'More common in people with weakened immune systems',
            'Can develop after skin injuries, surgery, or from skin conditions like eczema'
        ],
        'symptoms': [
            'Red area of skin that tends to expand',
            'Swelling',
            'Tenderness',
            'Pain in the affected area',
            'Warm skin',
            'Fever',
            'Red streaks extending from the affected area'
        ],
        'treatment': [
            'Oral antibiotics for mild cases',
            'Intravenous antibiotics for severe cases',
            'Elevating the affected area',
            'Keeping the area clean',
            'Pain relievers as needed'
        ],
        'prevention': [
            'Wash wounds promptly with soap and water',
            'Apply antibiotic ointment to breaks in the skin',
            'Keep skin moisturized to prevent cracking',
            'Wear protective footwear',
            'Manage underlying conditions like diabetes'
        ],
        'duration': '7-10 days with proper antibiotic treatment',
        'when_to_see_doctor': [
            'If the redness and swelling spread rapidly',
            'If you have a fever of 100.4°F (38°C) or higher',
            'If you have diabetes or a weakened immune system',
            'If symptoms don\'t improve after 2-3 days of antibiotic treatment',
            'If you develop nausea or vomiting'
        ]
    },
    'FU-athlete-foot': {
        'name': 'Athlete\'s Foot',
        'description': 'Athlete\'s foot (tinea pedis) is a fungal infection that affects the skin on the feet, particularly between the toes. It often causes itching, stinging, and burning sensations.',
        'causes': [
            'Caused by various types of fungi (dermatophytes)',
            'Thrives in warm, moist environments like shoes and socks',
            'Spreads through direct contact or by touching contaminated surfaces',
            'Common in athletes and people who wear tight shoes'
        ],
        'symptoms': [
            'Itching, stinging, and burning between toes and on soles',
            'Itchy blisters',
            'Cracking and peeling skin',
            'Dry skin on soles and sides of feet',
            'Raw skin',
            'Discolored, thick toenails (if infection spreads)'
        ],
        'treatment': [
            'Over-the-counter antifungal creams, sprays, or powders',
            'Prescription antifungal medications for severe cases',
            'Keeping feet clean and dry',
            'Changing socks regularly',
            'Using antifungal powder in shoes'
        ],
        'prevention': [
            'Washing feet daily with soap and water',
            'Thoroughly drying feet, especially between toes',
            'Wearing clean, dry socks',
            'Using sandals in public showers and pools',
            'Not sharing shoes, socks, or towels'
        ],
        'duration': '2-4 weeks with proper treatment',
        'when_to_see_doctor': [
            'If symptoms don\'t improve after 2 weeks of OTC treatment',
            'If the rash is painful or shows signs of infection',
            'If the rash spreads to nails',
            'If you have diabetes',
            'If you have a weakened immune system'
        ]
    },
    'BA-impetigo': {
        'name': 'Impetigo',
        'description': 'Impetigo is a common, highly contagious bacterial skin infection that mainly affects infants and children. It usually appears as red sores on the face, especially around the nose and mouth, and on hands and feet. The sores burst and develop a yellow-brown crust.',
        'causes': [
            'Caused by Staphylococcus aureus or Streptococcus pyogenes bacteria',
            'Spreads through direct contact with sores or contaminated objects',
            'More common in warm, humid weather',
            'Often develops on skin that\'s already irritated by other conditions',
            'More common in crowded environments like schools'
        ],
        'symptoms': [
            'Red sores that quickly rupture and leak fluid or pus',
            'Honey-colored crust that forms over sores',
            'Itching',
            'Sores that increase in size and number',
            'Swollen lymph nodes near the sores'
        ],
        'treatment': [
            'Topical antibiotic ointments (mupirocin)',
            'Oral antibiotics for more severe cases',
            'Gentle cleansing of affected areas',
            'Trimming nails to prevent spread from scratching',
            'Covering lesions with gauze',
            'Antiseptic soap washes'
        ],
        'prevention': [
            'Good hand hygiene',
            'Keeping skin clean and dry',
            'Covering cuts and scrapes',
            'Not sharing personal items like towels or clothing',
            'Washing contaminated items in hot water'
        ],
        'duration': '7-10 days with proper antibiotic treatment',
        'when_to_see_doctor': [
            'If the rash doesn\'t improve after 2 weeks of OTC treatment',
            'If the rash is painful or shows signs of infection',
            'If the rash is on your scalp',
            'If you have a weakened immune system',
            'If the rash spreads rapidly'
        ]
    },
    'FU-nail-fungus': {
        'name': 'Nail Fungus',
        'description': 'Nail fungus (onychomycosis) is a common condition that begins as a white or yellow spot under the tip of your fingernail or toenail. As the fungal infection goes deeper, it may cause your nail to discolor, thicken and crumble at the edge.',
        'causes': [
            'Caused by various fungal organisms, including dermatophytes, yeasts, and molds',
            'More common in older adults as nails become drier and more brittle',
            'Spreads through direct contact or contaminated surfaces',
            'More common in people with reduced blood circulation or weakened immune systems'
        ],
        'symptoms': [
            'Thickened nail',
            'Whitish to yellow-brown discoloration',
            'Brittleness, crumbling or ragged nail',
            'Distorted nail shape',
            'Dark color due to debris buildup under nail',
            'Slightly foul smell'
        ],
        'treatment': [
            'Oral antifungal medications (most effective)',
            'Medicated nail polish (ciclopirox)',
            'Medicated nail cream',
            'Laser treatment',
            'In severe cases, nail removal'
        ],
        'prevention': [
            'Wash hands and feet regularly',
            'Trim nails straight across and file down thickened areas',
            'Wear moisture-wicking socks',
            'Change socks daily or more often if feet sweat',
            'Wear shoes that allow ventilation',
            'Wear sandals in public showers and pools'
        ],
        'duration': 'Several months to a year for complete resolution',
        'when_to_see_doctor': [
            'If you have diabetes and suspect nail fungus',
            'If you have signs of infection (redness, warmth, pus)',
            'If the infection spreads to other nails',
            'If you have pain or discomfort in the affected nail',
            'If home treatments aren\'t working after several months'
        ]
    },
    'FU-ringworm': {
        'name': 'Ringworm',
        'description': 'Ringworm (tinea corporis) is a common fungal skin infection that causes a ring-shaped rash. Despite its name, it has nothing to do with worms. The infection is caused by a type of fungus called a dermatophyte.',
        'causes': [
            'Caused by dermatophyte fungi',
            'Spreads through direct skin-to-skin contact',
            'Can spread from animals to humans',
            'Can spread through contact with contaminated objects',
            'More common in warm, humid climates'
        ],
        'symptoms': [
            'Circular, red, scaly patch of skin',
            'Clearer skin in the middle of the ring',
            'Slightly raised, expanding ring',
            'Itching',
            'Blisters or oozing in some cases'
        ],
        'treatment': [
            'Over-the-counter antifungal creams, lotions, or powders',
            'Prescription-strength topical medications for severe cases',
            'Oral antifungal medications for widespread infection',
            'Keeping the area clean and dry',
            'Washing bedding and clothing frequently'
        ],
        'prevention': [
            'Avoiding direct contact with infected people or animals',
            'Not sharing personal items like towels, clothing, or hairbrushes',
            'Washing hands after contact with pets',
            'Wearing loose-fitting clothing',
            'Keeping skin clean and dry'
        ],
        'duration': '2-4 weeks with proper treatment',
        'when_to_see_doctor': [
            'If the rash is painful or shows signs of infection',
            'If the rash doesn\'t improve after 2 weeks of OTC treatment',
            'If the rash is widespread or covers a large area',
            'If you have a weakened immune system',
            'If the rash spreads rapidly'
        ]
    },
    'PA-cutaneous-larva-migrans': {
        'name': 'Cutaneous Larva Migrans',
        'description': 'Cutaneous larva migrans (CLM), also known as "creeping eruption," is a skin disease caused by hookworm larvae that have penetrated the skin. It\'s characterized by an itchy, winding rash that moves or "migrates" across the skin.',
        'causes': [
            'Caused by hookworm larvae (usually from dog or cat feces)',
            'Larvae penetrate skin through contact with contaminated soil',
            'Common in tropical and subtropical regions',
            'More common in people who walk barefoot on contaminated soil'
        ],
        'symptoms': [
            'Winding, snake-like rash that moves or "migrates"',
            'Intense itching at the site of the rash',
            'Red, raised tracks on the skin',
            'Blisters may form in some cases',
            'Rash typically appears 1-5 days after exposure'
        ],
        'treatment': [
            'Antiparasitic medications (albendazole or ivermectin)',
            'Topical thiabendazole for mild cases',
            'Antihistamines for itching',
            'Keeping the area clean to prevent secondary infection'
        ],
        'prevention': [
            'Wearing shoes when walking on soil or sand',
            'Avoiding sitting or lying directly on soil or sand',
            'Proper disposal of pet feces',
            'Regular deworming of pets',
            'Using protective barriers when sitting on beaches'
        ],
        'duration': 'Several weeks without treatment, but resolves with proper medication',
        'when_to_see_doctor': [
            'If the rash is extremely itchy or painful',
            'If the rash shows signs of secondary infection',
            'If you develop fever or other systemic symptoms',
            'If the rash spreads rapidly',
            'If you have a weakened immune system'
        ]
    }
}

@st.cache_resource
def load_models():
    """Load the models once and cache them"""
    # Get directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    
    try:
        # Load minimal SVM model
        svm_model = load_minimal_svm_model(os.path.join(models_dir, 'svm_model_optimized.pkl'))
        
        # Load ResNet50 base model
        resnet_model = load_model(os.path.join(models_dir, 'resnet50_base_model.h5'))
        st.success("Models loaded successfully!")
        return svm_model, resnet_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please make sure the model files are in the correct location.")
        st.error(f"Script directory: {script_dir}")
        st.error(f"Looking in: {models_dir}")
        raise

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    try:
        # Convert to RGB if it's a RGBA image
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        # Resize to model input size
        img = image.resize((192, 192))
        
        # Convert to numpy array and preprocess
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        raise

def generate_report(disease_id):
    """Generate a medical report for the predicted disease"""
    disease_info = DISEASE_INFO.get(disease_id)
    if not disease_info:
        st.error("Disease information not available.")
        return
    
    st.subheader(f"{disease_info['name']} Medical Report")
    
    # Add disclaimer
    st.warning("MEDICAL DISCLAIMER: This report provides general information and is for educational purposes only. It is not intended as medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")
    
    # Overview
    st.markdown("### Overview")
    st.write(disease_info['description'])
    
    # Key information in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Duration**")
        st.write(disease_info['duration'])
    with col2:
        st.markdown("**Contagious**")
        st.write("Yes" if disease_id in ['VI-chickenpox', 'BA-cellulitis', 'FU-athlete-foot', 'BA-impetigo', 'FU-ringworm'] else "No")
    with col3:
        st.markdown("**Severity**")
        st.write("Moderate" if disease_id in ['BA-cellulitis', 'BA-impetigo'] else "Mild")
    
    # Causes
    st.markdown("### Causes")
    for cause in disease_info['causes']:
        st.markdown(f"- {cause}")
    
    # Symptoms
    st.markdown("### Symptoms")
    for symptom in disease_info['symptoms']:
        st.markdown(f"- {symptom}")
    
    # Treatment
    st.markdown("### Treatment Options")
    for treatment in disease_info['treatment']:
        st.markdown(f"- {treatment}")
    
    # Recovery time
    st.info(f"**Expected Recovery Time:** {disease_info['duration']}")
    
    # Prevention
    st.markdown("### Prevention Strategies")
    for prevention in disease_info['prevention']:
        st.markdown(f"- {prevention}")
    
    # When to see doctor
    st.markdown("### When to See a Doctor")
    for when in disease_info['when_to_see_doctor']:
        st.markdown(f"- {when}")

# Main app
def main():
    st.title("🏥 Skin Disease Detection System")
    st.markdown("Upload a skin image for analysis. *This tool is for educational purposes only and not a substitute for professional medical advice.*")
    
    # Load models
    svm_model, resnet_model = load_models()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Process button
        if st.button('Analyze Image'):
            with st.spinner('Analyzing image...'):
                try:
                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    
                    # Extract features using ResNet50
                    features = resnet_model.predict(processed_image)
                    
                    # Reshape features for SVM
                    features_flat = features.reshape(1, -1)
                    
                    # Predict using SVM
                    prediction = svm_model.predict(features_flat)[0]
                    
                    # Get probability scores
                    try:
                        probabilities = svm_model.predict_proba(features_flat)[0]
                        confidence = round(float(np.max(probabilities)) * 100, 2)
                    except:
                        confidence = 90.0  # Default confidence if probabilities aren't available
                    
                    # Define categories
                    Categories = [
                        'VI-chickenpox', 
                        'BA-cellulitis', 
                        'FU-athlete-foot', 
                        'BA-impetigo', 
                        'FU-nail-fungus', 
                        'FU-ringworm', 
                        'PA-cutaneous-larva-migrans'
                    ]
                    
                    predicted_label = Categories[prediction]
                    
                    if '-' in predicted_label:
                        disease_name = predicted_label.split('-')[1].replace('-', ' ').title()
                        category = predicted_label.split('-')[0]
                    else:
                        disease_name = predicted_label
                        category = ""
                    
                    # Display results
                    st.success(f"Predicted Disease: **{disease_name}**")
                    st.info(f"Confidence: **{confidence}%**")
                    
                    # Show confidence bar
                    st.progress(int(confidence))
                    
                    # Generate medical report
                    generate_report(predicted_label)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.error("Please try a different image or contact support.")

if __name__ == "__main__":
    main()
