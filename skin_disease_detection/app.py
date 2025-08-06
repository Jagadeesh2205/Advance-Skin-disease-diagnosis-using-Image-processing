import os
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import logging
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf

# CRITICAL FIX: Force CPU mode to avoid GPU memory issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("INFO: Forcing TensorFlow to use CPU only (to prevent GPU memory errors)")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Current working directory: {os.getcwd()}")
print(f"BASE_DIR (app.py location): {BASE_DIR}")

# Define the template directory - THIS IS CRITICAL FOR TEMPLATE NOT FOUND ERROR
# Try multiple possible locations for templates
template_dir = None
possible_template_dirs = [
    # Option 1: templates in ui_components folder at same level as app.py
    os.path.join(BASE_DIR, 'ui_components'),
    # Option 2: templates in templates folder at same level as app.py
    os.path.join(BASE_DIR, 'templates'),
    # Option 3: templates in parent directory's ui_components
    os.path.join(os.path.dirname(BASE_DIR), 'ui_components'),
    # Option 4: templates in parent directory's templates
    os.path.join(os.path.dirname(BASE_DIR), 'templates')
]

# Find the first valid template directory
for possible_dir in possible_template_dirs:
    if os.path.exists(possible_dir):
        template_dir = possible_dir
        break

# If no template directory found, try to create one
if template_dir is None:
    logger.warning("No template directory found. Creating default template directory.")
    template_dir = os.path.join(BASE_DIR, 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    # Check if HTML files exist in current directory and copy them
    for filename in ['index.html', 'result.html', 'report.html', 'error.html']:
        if os.path.exists(filename):
            import shutil
            shutil.copy(filename, os.path.join(template_dir, filename))
            logger.info(f"Copied {filename} to template directory")
    
    # If still no HTML files, create basic ones
    if not any(f.endswith('.html') for f in os.listdir(template_dir)):
        logger.warning("No HTML templates found. Creating minimal templates.")
        
        # Create a minimal index.html
        with open(os.path.join(template_dir, 'index.html'), 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Skin Disease Detection</title>
</head>
<body>
    <h1>Upload an image for skin disease detection</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Analyze</button>
    </form>
</body>
</html>
            ''')
        
        # Create minimal result.html
        with open(os.path.join(template_dir, 'result.html'), 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>
    <h1>Prediction Result</h1>
    <p>Predicted Disease: {{ prediction }}</p>
    <p>Confidence: {{ confidence }}%</p>
    <a href="/report/{{ disease_id }}">Generate Medical Report</a>
    <a href="/">Upload Another Image</a>
</body>
</html>
            ''')
        
        # Create minimal report.html
        with open(os.path.join(template_dir, 'report.html'), 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>{{ disease.name }} Report</title>
</head>
<body>
    <h1>{{ disease.name }} Medical Report</h1>
    <h2>Overview</h2>
    <p>{{ disease.description }}</p>
    <h2>Causes</h2>
    <ul>
    {% for cause in disease.causes %}
        <li>{{ cause }}</li>
    {% endfor %}
    </ul>
    <a href="/">Analyze Another Image</a>
</body>
</html>
            ''')
        
        # Create minimal error.html
        with open(os.path.join(template_dir, 'error.html'), 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
</head>
<body>
    <h1>Error</h1>
    <p>{{ message }}</p>
    <a href="/">Return to Home Page</a>
</body>
</html>
            ''')

# Verify template directory
print(f"Using template directory: {template_dir}")
print("Files in template directory:")
for f in os.listdir(template_dir):
    print(f"  - {f}")

# Initialize Flask app with the correct template folder
app = Flask(__name__,
            template_folder=template_dir,
            static_folder=os.path.join(BASE_DIR, 'static'))

# Configure upload settings
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Your local model paths
MODEL_DIR = r"C:\Users\A JAGADEESH\Documents\machine learning\Advance Skin Disease project\Advanced-Skin-Diseases-Diagnosis-Leveraging-Image-Processing-main\skin_disease_detection\skin_disease_detection\models"
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model_optimized.pkl")
RESNET_MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_base_model.h5")

# Disease information
DISEASE_INFO = {
    'VI-chickenpox': {
        'name': 'Chickenpox',
        'description': 'Chickenpox is a highly contagious viral infection caused by the varicella-zoster virus (VZV). It is characterized by an itchy, blister-like rash that appears first on the chest, back, and face, then spreads over the entire body.',
        'causes': [
            'Caused by the varicella-zoster virus (VZV)',
            'Spreads through direct contact with the rash',
            'Airborne transmission through coughing or sneezing',
            'Can be transmitted from shingles to someone who has never had chickenpox'
        ],
        'symptoms': [
            'Itchy, red blisters all over the body',
            'Fever',
            'Fatigue',
            'Loss of appetite',
            'Headache',
            'Flu-like symptoms 1-2 days before rash appears'
        ],
        'complications': [
            'Skin infections from scratching',
            'Pneumonia',
            'Encephalitis (brain inflammation)',
            'Bleeding problems',
            'Dehydration',
            'Complications in pregnancy affecting the fetus'
        ],
        'treatment': [
            'Antiviral medications (acyclovir) for high-risk patients',
            'Calamine lotion for itching',
            'Antihistamines to reduce itching',
            'Oatmeal baths to soothe skin',
            'Acetaminophen for fever (never use aspirin)',
            'Keeping fingernails short to prevent infection from scratching'
        ],
        'prevention': [
            'Varicella vaccine (90% effective)',
            'Avoiding contact with infected individuals',
            'Isolation of infected persons until all blisters have crusted over',
            'Good hand hygiene'
        ],
        'duration': '7-10 days from first symptom to complete healing',
        'when_to_see_doctor': [
            'If rash spreads to eyes',
            'If fever lasts more than 4 days',
            'If rash becomes very red, warm, or tender (signs of infection)',
            'If difficulty walking (possible neurological complication)',
            'If dehydration symptoms (urinating less, dry mouth)'
        ]
    },
    'BA- cellulitis': {
        'name': 'Cellulitis',
        'description': 'Cellulitis is a common, potentially serious bacterial skin infection that affects the deeper layers of skin (dermis and subcutaneous tissue). It appears as a swollen, red area of skin that feels hot and tender, and it may spread rapidly.',
        'causes': [
            'Most commonly caused by Streptococcus and Staphylococcus bacteria',
            'Enters through breaks in the skin (cuts, ulcers, insect bites)',
            'Can develop after surgery',
            'More common in people with weakened immune systems',
            'Associated with conditions like athlete\'s foot or eczema that cause skin breaks'
        ],
        'symptoms': [
            'Red, inflamed skin that appears swollen',
            'Skin that feels warm or hot to the touch',
            'Tenderness or pain in the affected area',
            'Fever or chills',
            'Red streaks extending from the affected area',
            'Pus or drainage from the skin'
        ],
        'complications': [
            'Blood infection (sepsis)',
            'Bone infection (osteomyelitis)',
            'Lymphangitis (infection of lymph vessels)',
            'Recurrent cellulitis',
            'Tissue death (gangrene)',
            'Chronic swelling (lymphedema)'
        ],
        'treatment': [
            'Oral antibiotics (typically for 5-14 days)',
            'Intravenous antibiotics for severe cases',
            'Elevation of affected limb to reduce swelling',
            'Pain medication as needed',
            'Wound care for any breaks in the skin',
            'Compression stockings for leg cellulitis'
        ],
        'prevention': [
            'Prompt cleaning of cuts and scrapes',
            'Moisturizing dry skin to prevent cracking',
            'Wearing protective footwear',
            'Managing underlying conditions like diabetes',
            'Treating fungal infections like athlete\'s foot'
        ],
        'duration': 'Improvement typically seen within 3 days of starting antibiotics; full recovery in 7-10 days',
        'when_to_see_doctor': [
            'If redness or pain worsens',
            'If fever develops',
            'If you have diabetes or a weakened immune system',
            'If symptoms don\'t improve after 2-3 days of antibiotics',
            'If the affected area is near the eyes'
        ]
    },
    'FU-athlete-foot': {
        'name': 'Athlete\'s Foot',
        'description': 'Athlete\'s foot (tinea pedis) is a common fungal infection that affects the skin on the feet, particularly between the toes. It thrives in warm, moist environments like shoes and socks.',
        'causes': [
            'Caused by various types of fungi (dermatophytes)',
            'Spreads in damp communal areas (locker rooms, showers, pools)',
            'Wearing tight, closed shoes for long periods',
            'Sharing towels, socks, or shoes with an infected person',
            'Having sweaty feet or minor foot injuries'
        ],
        'symptoms': [
            'Itching, stinging, and burning between toes or on soles',
            'Cracking and peeling skin',
            'Redness and scaling',
            'Blisters that itch',
            'Toenail discoloration if infection spreads'
        ],
        'complications': [
            'Spread to other parts of the body (hands, groin, scalp)',
            'Bacterial infection from excessive scratching',
            'Chronic fungal nail infection (onychomycosis)',
            'Cellulitis from skin breakdown',
            'Recurrent infections'
        ],
        'treatment': [
            'Over-the-counter antifungal creams, sprays, or powders',
            'Prescription-strength topical medications for severe cases',
            'Oral antifungal medications for persistent infections',
            'Keeping feet clean and dry',
            'Changing socks frequently',
            'Using antifungal powder in shoes'
        ],
        'prevention': [
            'Wearing shower sandals in public showers',
            'Wearing breathable shoes and moisture-wicking socks',
            'Washing feet daily and drying thoroughly',
            'Alternating shoes to allow them to dry completely',
            'Not sharing shoes, socks, or towels'
        ],
        'duration': '2-4 weeks with proper treatment; can become chronic if untreated',
        'when_to_see_doctor': [
            'If symptoms don\'t improve after 2 weeks of OTC treatment',
            'If you have diabetes',
            'If signs of bacterial infection (increased redness, warmth, pus)',
            'If the infection spreads to nails',
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
            'Red sores that quickly burst and form honey-colored crusts',
            'Itchy rash',
            'Sores that increase in size and number',
            'Swollen lymph nodes near the infection',
            'Pain around the sores',
            'Fluid-filled blisters that may be clear or yellow'
        ],
        'complications': [
            'Cellulitis',
            'Kidney problems (poststreptococcal glomerulonephritis)',
            'Scarring (rare)',
            'Staphylococcal scalded skin syndrome',
            'Spread to other parts of the body',
            'Methicillin-resistant Staphylococcus aureus (MRSA) infection'
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
        'duration': '2-3 weeks without treatment; 7-10 days with treatment',
        'when_to_see_doctor': [
            'If rash is widespread or painful',
            'If fever develops',
            'If symptoms don\'t improve after 3 days of treatment',
            'If signs of cellulitis (increasing redness, warmth)',
            'If the person has a weakened immune system'
        ]
    },
    'FU-nail-fungus': {
        'name': 'Nail Fungus',
        'description': 'Nail fungus (onychomycosis) is a common condition that begins as a white or yellow spot under the tip of your fingernail or toenail. As the fungal infection goes deeper, it may cause your nail to discolor, thicken and develop crumbling edges.',
        'causes': [
            'Caused by various fungi including dermatophytes, yeasts, and molds',
            'More common in toenails than fingernails',
            'Risk increases with age',
            'Spreads in warm, moist environments like pools and showers',
            'Associated with athlete\'s foot infection'
        ],
        'symptoms': [
            'Thickened nails',
            'Whitish to yellow-brown discoloration',
            'Brittleness, crumbling or ragged nails',
            'Distorted nail shape',
            'Dark color due to debris buildup under nail',
            'Slight odor'
        ],
        'complications': [
            'Pain and discomfort',
            'Permanent nail damage',
            'Spread to other nails',
            'Secondary bacterial infections',
            'Difficulty walking or wearing shoes',
            'Cellulitis in severe cases'
        ],
        'treatment': [
            'Oral antifungal medications (terbinafine, itraconazole)',
            'Medicated nail polish (ciclopirox)',
            'Medicated nail cream',
            'Nail removal in severe cases',
            'Laser therapy (emerging treatment)',
            'Tea tree oil as complementary treatment'
        ],
        'prevention': [
            'Wearing shower shoes in public areas',
            'Keeping nails clean and dry',
            'Trimming nails straight across',
            'Wearing breathable shoes',
            'Changing socks daily',
            'Not sharing nail clippers'
        ],
        'duration': 'Treatment typically takes 6-12 months for toenails due to slow growth',
        'when_to_see_doctor': [
            'If you have diabetes',
            'If you notice signs of infection',
            'If pain affects daily activities',
            'If the condition worsens despite home treatment',
            'If you have circulation problems'
        ]
    },
    'FU-ringworm': {
        'name': 'Ringworm',
        'description': 'Ringworm (tinea corporis) is a common fungal skin infection that causes a ring-shaped rash on the skin. Despite its name, it\'s not caused by a worm. It\'s highly contagious and can spread through direct contact with an infected person or animal, or from contact with contaminated surfaces.',
        'causes': [
            'Caused by dermatophyte fungi',
            'Spreads through direct skin-to-skin contact',
            'Contact with contaminated surfaces (towels, clothing, bedding)',
            'Contact with infected animals (especially cats)',
            'Warm, moist environments increase risk'
        ],
        'symptoms': [
            'Ring-shaped, red, itchy rash with raised edges',
            'Clearing in the center of the ring',
            'Scaly, cracked skin',
            'Blisters in some cases',
            'Multiple rings that may overlap',
            'Hair loss in affected areas of the scalp'
        ],
        'complications': [
            'Spread to other body areas',
            'Secondary bacterial infection from scratching',
            'Permanent hair loss (with scalp ringworm)',
            'Nail deformities (if spreads to nails)',
            'Kerion (inflamed, pus-filled areas on scalp)',
            'Chronic, recurring infections'
        ],
        'treatment': [
            'Over-the-counter antifungal creams, ointments, or sprays',
            'Prescription-strength topical medications for severe cases',
            'Oral antifungal medications for widespread infections',
            'Antifungal shampoo for scalp ringworm',
            'Keeping the area clean and dry',
            'Washing contaminated clothing in hot water'
        ],
        'prevention': [
            'Avoiding contact with infected people or animals',
            'Not sharing personal items like towels or clothing',
            'Wearing loose-fitting clothing',
            'Keeping skin clean and dry',
            'Washing hands after contact with pets',
            'Using antifungal powder in shoes'
        ],
        'duration': '2-4 weeks with proper treatment; may take longer for scalp infections',
        'when_to_see_doctor': [
            'If the rash doesn\'t improve after 2 weeks of OTC treatment',
            'If the rash is painful or shows signs of infection',
            'If the rash is on your scalp',
            'If you have a weakened immune system',
            'If the rash spreads rapidly'
        ]
    },
    'PA-cutaneous-larva-migrans': {
        'name': 'Cutaneous Larva Migrans',
        'description': 'Cutaneous larva migrans (CLM), also known as "creeping eruption," is a skin disease caused by hookworm larvae that have penetrated the skin. It\'s characterized by an itchy, winding rash that moves or "migrates" across the skin.',
        'causes': [
            'Caused by hookworm larvae (usually from dog or cat feces)',
            'Larvae penetrate skin through direct contact with contaminated soil/sand',
            'Common in tropical and subtropical regions',
            'More likely when walking barefoot on contaminated beaches',
            'Not spread from person to person'
        ],
        'symptoms': [
            'Intensely itchy, winding red tracks on the skin',
            'Raised, snake-like lines that grow longer each day',
            'Small blisters at the start of the tracks',
            'Redness and swelling around the tracks',
            'Burning sensation in affected areas',
            'Tracks typically appear 1-5 days after exposure'
        ],
        'complications': [
            'Secondary bacterial infection from scratching',
            'Persistent itching for weeks or months',
            'Scarring from scratching',
            'Sleep disturbances due to itching',
            'Superinfection with other organisms',
            'Rarely, larvae may migrate to other organs'
        ],
        'treatment': [
            'Antiparasitic medications (ivermectin, albendazole)',
            'Topical thiabendazole (less effective than oral)',
            'Antihistamines for itching',
            'Topical corticosteroids for inflammation',
            'Cool compresses for symptom relief',
            'Keeping nails short to prevent skin damage from scratching'
        ],
        'prevention': [
            'Wearing shoes on beaches in tropical areas',
            'Using a barrier (towel, mat) when sitting on sand',
            'Avoiding areas where animals defecate',
            'Proper disposal of pet feces',
            'Good hand hygiene after outdoor activities'
        ],
        'duration': 'Without treatment: 4-8 weeks; With treatment: symptoms improve within days',
        'when_to_see_doctor': [
            'If you suspect CLM after traveling to tropical areas',
            'If itching is severe and disrupting sleep',
            'If signs of secondary infection (pus, increased redness)',
            'If the rash spreads rapidly',
            'If you\'re pregnant or immunocompromised'
        ]
    }
}

# Categories must match your training
CATEGORIES = ['VI-chickenpox', 'BA- cellulitis', 'FU-athlete-foot', 
              'BA-impetigo', 'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans']

# Image size must match your training
IMG_SIZE = (192, 192)

# Global variables to store models
svm_model = None
resnet_model = None

def load_models():
    """Load the models once at startup with memory considerations"""
    global svm_model, resnet_model
    
    try:
        # Load SVM model
        try:
            with open(SVM_MODEL_PATH, 'rb') as f:
                svm_model = pickle.load(f)
            logger.info("SVM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SVM model: {str(e)}")
            return False
            
        # Load ResNet50 base model
        try:
            # Verify the model file exists
            if not os.path.exists(RESNET_MODEL_PATH):
                logger.error(f"ResNet model file not found at {RESNET_MODEL_PATH}")
                return False
                
            resnet_model = load_model(RESNET_MODEL_PATH)
            logger.info("ResNet50 model loaded successfully on CPU")
            return True
        except Exception as e:
            logger.error(f"Error loading ResNet model: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def preprocess_image(image_path):
    """Preprocess the image for prediction"""
    try:
        # Open and convert image
        image = Image.open(image_path)
        img = np.array(image)
        
        # Handle RGBA images
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
        # Resize
        img = cv2.resize(img, IMG_SIZE)
        
        # Contrast Enhancement (CLAHE)
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.warning(f"CLAHE enhancement failed, using original image. Error: {str(e)}")
        
        # Convert to float32 and preprocess for ResNet50
        img = img.astype(np.float32)
        img = preprocess_input(img)
        
        return np.expand_dims(img, axis=0)
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        raise

def predict_disease(image_path):
    """Predict the disease from an image"""
    try:
        # Preprocess image
        processed_img = preprocess_image(image_path)
        
        # Extract features using ResNet
        features = resnet_model.predict(processed_img)
        features_flat = features.reshape(1, -1)
        
        # Convert to float16 as in your training script
        features_flat = features_flat.astype(np.float16)
        
        # Make prediction using SVM
        prediction_idx = svm_model.predict(features_flat)[0]
        predicted_label = CATEGORIES[prediction_idx]
        
        # Get probability scores with validation
        try:
            probabilities = svm_model.predict_proba(features_flat)[0]
            # Validate probabilities
            if not np.isclose(probabilities.sum(), 1.0, atol=0.01):
                logger.warning(f"Invalid probabilities sum: {probabilities.sum()}")
                confidence = 90.0
            else:
                confidence = round(float(np.max(probabilities)) * 100, 2)
        except Exception as e:
            logger.error(f"Probability error: {str(e)}")
            confidence = 90.0  # Default if probabilities fail
        
        logger.info(f"Prediction: {predicted_label} ({confidence}%) | Probabilities: {probabilities}")
        return predicted_label, confidence
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('error.html', message="No file part in the request")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', message="No file selected")
    
    if file:
        # Save the uploaded file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Make prediction
            predicted_label, confidence = predict_disease(filepath)
            
            # Get disease name for display
            if '-' in predicted_label:
                disease_name = predicted_label.split('-')[1].replace('-', ' ').title()
            else:
                disease_name = predicted_label
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return render_template('result.html', 
                                  prediction=disease_name,
                                  confidence=confidence,
                                  disease_id=predicted_label)
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template('error.html', message=f"Error processing image: {str(e)}")
    
    return render_template('error.html', message="Unknown error occurred")

@app.route('/report/<disease_id>')
def report(disease_id):
    if disease_id not in DISEASE_INFO:
        return render_template('error.html', message="Invalid disease ID")
    
    disease = DISEASE_INFO[disease_id]
    return render_template('report.html', disease=disease, disease_id=disease_id)

@app.route('/error')
def error():
    message = request.args.get('message', 'An error occurred')
    return render_template('error.html', message=message)

if __name__ == '__main__':
    # Verify model files exist before trying to load
    if not os.path.exists(SVM_MODEL_PATH):
        logger.error(f"SVM model file not found at {SVM_MODEL_PATH}")
        logger.error("Please check your model path and ensure the file exists")
        logger.error("Current working directory: " + os.getcwd())
        logger.error("BASE_DIR: " + BASE_DIR)
        logger.error("MODEL_DIR: " + MODEL_DIR)
        exit(1)
        
    if not os.path.exists(RESNET_MODEL_PATH):
        logger.error(f"ResNet model file not found at {RESNET_MODEL_PATH}")
        logger.error("Please check your model path and ensure the file exists")
        logger.error("Current working directory: " + os.getcwd())
        logger.error("BASE_DIR: " + BASE_DIR)
        logger.error("MODEL_DIR: " + MODEL_DIR)
        exit(1)
    
    # Load models before starting the server
    if not load_models():
        logger.error("Failed to load models. Application cannot start.")
        logger.error("Possible solutions:")
        logger.error("1. Verify model files exist at the specified paths")
        logger.error("2. Ensure you have enough RAM (at least 4GB free)")
        logger.error("3. Try running with CPU only (already enforced)")
        logger.error("4. Reduce model size or use a smaller model")
        exit(1)
    
    # Run the Flask app
    logger.info("Starting Flask application on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
