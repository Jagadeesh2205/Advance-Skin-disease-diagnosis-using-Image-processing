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
    # Add similar detailed information for the other 5 skin conditions
    # I'll provide examples for 2 more, but you should complete all 7
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
    # You should complete the remaining 3 diseases with similar detailed information
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