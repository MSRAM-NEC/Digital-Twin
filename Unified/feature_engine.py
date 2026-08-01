import numpy as np
import re

# Simple keyword spotting for typical activities
ACTIVITIES = {
    'run': {'intensity': 0.8, 'base_hr': 140},
    'running': {'intensity': 0.8, 'base_hr': 140},
    'swim': {'intensity': 0.75, 'base_hr': 130},
    'swimming': {'intensity': 0.75, 'base_hr': 130},
    'walk': {'intensity': 0.3, 'base_hr': 90},
    'walking': {'intensity': 0.3, 'base_hr': 90},
    'sleep': {'intensity': 0.0, 'base_hr': 55},
    'sleeping': {'intensity': 0.0, 'base_hr': 55},
    'chess': {'intensity': 0.1, 'base_hr': 75},
    'gym': {'intensity': 0.6, 'base_hr': 120},
    'workout': {'intensity': 0.6, 'base_hr': 120},
    'cycling': {'intensity': 0.7, 'base_hr': 125},
    'bike': {'intensity': 0.7, 'base_hr': 125},
}

def classify_activity(query):
    query = query.lower()
    best_match = 'walking' # default
    for act in ACTIVITIES:
        if act in query:
            # Prefer longer matches if possible
            if len(act) > len(best_match) or best_match == 'walking':
                best_match = act
                
    if best_match == 'run': return 'running'
    if best_match == 'swim': return 'swimming'
    if best_match == 'walk': return 'walking'
    if best_match == 'bike': return 'cycling'
    if best_match == 'workout': return 'gym'
    return best_match

def get_activity_name(query):
    act = classify_activity(query)
    return act.capitalize()

def estimate_heart_rate(activity_id, config):
    act_info = ACTIVITIES.get(activity_id, ACTIVITIES['walking'])
    
    # Karvonen Formula Est.
    age = config.get('age', 45)
    resting_hr = config.get('resting_hr', 70)
    max_hr = 220 - age
    hr_reserve = max_hr - resting_hr
    
    intensity = act_info['intensity']
    target_hr = resting_hr + (hr_reserve * intensity)
    return target_hr

def build_feature_vector(query, config):
    # Standardize 22 length feature vector for the Scaler/ML Models
    # Since we need to trigger dynamic variations based on the action,
    # we bind the numerical features explicitly:
    f = np.zeros(22)
    f[0] = config.get('age', 45)
    f[1] = config.get('height_cm', 175)
    f[2] = config.get('weight_kg', 75)
    f[3] = config.get('resting_hr', 70)
    
    # Feature engineering metrics
    act = classify_activity(query)
    intensity = ACTIVITIES.get(act, ACTIVITIES['walking'])['intensity']
    f[4] = intensity
    
    # Extract optional duration (e.g. "for 30 minutes")
    duration = 30 # default
    nums = re.findall(r'\d+', query)
    if nums:
        duration = int(nums[-1])
    f[5] = duration
    
    # One-hot encoded activity padding (using modulo hash to stabilize inputs per activity type)
    act_index = sum(ord(c) for c in act) % 15
    f[6 + act_index] = 1.0

    return f
