"""
app/services/advice_service.py
Provides scientifically grounded, actionable agricultural and disease management advice.
Addresses Project Objective 4: 'What general management advice should the user consider?'
"""

DISEASE_ADVICE = {
    "Pepper__bell___Bacterial_spot": {
        "title": "Bacterial Spot in Bell Pepper",
        "severity": "High",
        "symptoms": "Small, yellow-green spots on leaves that turn brown with water-soaked margins; fruit spots can cause rot.",
        "cultural_control": [
            "Avoid overhead irrigation to minimize leaf wetness duration.",
            "Remove and destroy infected crop residues immediately after harvest.",
            "Rotate peppers with non-solanaceous crops (e.g., legumes, cereals) for at least 2 years."
        ],
        "organic_treatment": [
            "Apply preventive copper-based bactericides (copper hydroxide) during early vegetation before heavy rains.",
            "Utilize bio-fungicides containing Bacillus subtilis or Trichoderma species to suppress bacterial colonization."
        ],
        "prevention": "Use certified disease-free seeds and resistant varieties if available."
    },
    "Pepper__bell___healthy": {
        "title": "Healthy Bell Pepper Leaf",
        "severity": "None",
        "symptoms": "Vibrant green foliage with no detectable lesion or discoloration.",
        "cultural_control": [
            "Maintain balanced N-P-K fertigation and ensure adequate drainage.",
            "Mulch around plants to retain moisture and prevent soil-borne pathogens from splashing onto lower leaves."
        ],
        "organic_treatment": [
            "Routine foliar seaweed extract or compost tea spray to bolster systemic acquired resistance (SAR)."
        ],
        "prevention": "Inspect leaves weekly, particularly the undersides, for early signs of aphids or thrips."
    },
    "Potato___Early_blight": {
        "title": "Early Blight in Potato (Alternaria solani)",
        "severity": "Medium",
        "symptoms": "Concentric 'target board' dark brown to black rings on older foliage, progressing upwards.",
        "cultural_control": [
            "Avoid sprinkler irrigation; use drip lines or furrow watering.",
            "Promote good plant spacing and aeration to speed foliage drying.",
            "Maintain plant vigor; nutrient-stressed potatoes are significantly more susceptible."
        ],
        "organic_treatment": [
            "Apply copper sulfate or fixed copper sprays every 7-10 days during warm, humid periods.",
            "Neem oil spray can provide secondary antifungal deterrence."
        ],
        "prevention": "Implement a 3-4 year crop rotation away from solanaceous plants (tomatoes, peppers, eggplants)."
    },
    "Potato___Late_blight": {
        "title": "Late Blight in Potato (Phytophthora infestans)",
        "severity": "Critical",
        "symptoms": "Rapidly expanding water-soaked lesions that turn dark purplish-black with white mildew growth underneath during wet weather.",
        "cultural_control": [
            "Destroy cull piles and volunteer potato sprouts where the pathogen overwinters.",
            "Hill soil well over tuber beds to prevent zoospores from washing down into tubers.",
            "Remove and bag severely infected plants during dry sunny weather."
        ],
        "organic_treatment": [
            "Strict preventative copper hydroxide applications prior to prolonged humid or rainy spells.",
            "Bio-rational treatments with phosphonates can stimulate plant defenses."
        ],
        "prevention": "Plant certified disease-free seed tubers and select late blight resistant cultivars."
    },
    "Potato___healthy": {
        "title": "Healthy Potato Leaf",
        "severity": "None",
        "symptoms": "Uniform green foliage, firm stems, healthy development.",
        "cultural_control": [
            "Ensure regular hilling to protect developing tubers from sun and pests.",
            "Monitor soil moisture carefully during tuber initiation and bulking."
        ],
        "organic_treatment": [
            "Maintain regular preventive scouting and beneficial insect habitats."
        ],
        "prevention": "Maintain strict sanitation of farm tools and machinery."
    },
    "Tomato_Bacterial_spot": {
        "title": "Bacterial Spot in Tomato (Xanthomonas spp.)",
        "severity": "High",
        "symptoms": "Dark, angular, water-soaked leaf lesions surrounded by yellow halos; foliage may become ragged.",
        "cultural_control": [
            "Never cultivate or harvest when tomato plants are wet with dew or rain.",
            "Disinfect stakes, cages, and pruning shears with a 10% bleach or sanitizing solution.",
            "Use drip irrigation to keep water off tomato foliage."
        ],
        "organic_treatment": [
            "Foliar sprays of copper-mancozeb tank mixes or bio-fungicides with Bacillus amyloliquefaciens."
        ],
        "prevention": "Ensure hot water seed treatment and purchase certified disease-free transplants."
    },
    "Tomato_Early_blight": {
        "title": "Early Blight in Tomato (Alternaria linariae)",
        "severity": "Medium",
        "symptoms": "Concentric dark brown rings on lower leaves, causing premature defoliation from the base upwards.",
        "cultural_control": [
            "Prune off the bottom 12 inches of foliage once plants are established to eliminate soil contact.",
            "Apply straw, plastic, or paper mulch to prevent rain splashing soil onto foliage.",
            "Stake or cage tomatoes to keep growth upright and improve airflow."
        ],
        "organic_treatment": [
            "Preventive spray with copper soap or sulfur compounds upon first sign of lower leaf spotting.",
            "Bio-fungicides based on Streptomyces lydicus or Trichoderma harzianum."
        ],
        "prevention": "Rotate garden beds on a 3-year minimum cycle."
    },
    "Tomato_Late_blight": {
        "title": "Late Blight in Tomato (Phytophthora infestans)",
        "severity": "Critical",
        "symptoms": "Large, greasy-looking grayish-green to dark brown blotches on leaves and fruit with fuzzy white fungal spores under humid conditions.",
        "cultural_control": [
            "Inspect garden daily during cool (15-22°C), humid, rainy periods.",
            "Promptly pull out and securely dispose of heavily infected vines (do NOT compost).",
            "Keep foliage completely dry with undercover cultivation (high tunnels/greenhouses) if feasible."
        ],
        "organic_treatment": [
            "Preventative copper-based fungicides applied before anticipated wet weather fronts."
        ],
        "prevention": "Select resistant varieties (e.g., Defiant Ph-R, Mountain Merit, Iron Lady)."
    },
    "Tomato_Leaf_Mold": {
        "title": "Tomato Leaf Mold (Passalora fulva)",
        "severity": "Medium",
        "symptoms": "Pale greenish-yellow spots on upper leaf surfaces corresponding to olive-green to brown velvety molds on undersides.",
        "cultural_control": [
            "Common in greenhouse/high-tunnel settings; increase passive ventilation and air circulation.",
            "Maintain relative humidity below 85% with spacing and morning watering."
        ],
        "organic_treatment": [
            "Apply copper hydroxide or potassium bicarbonate sprays to undersides of leaves."
        ],
        "prevention": "Choose resistant tomato hybrids carrying Cf resistance genes."
    },
    "Tomato_Septoria_leaf_spot": {
        "title": "Septoria Leaf Spot (Septoria lycopersici)",
        "severity": "Medium",
        "symptoms": "Numerous circular spots (1-3mm) with dark borders and tan or gray centers containing tiny black specks.",
        "cultural_control": [
            "Remove diseased lower leaves as soon as spots are noticed.",
            "Mulch thoroughly to suppress splash-borne spores from overwintering ground debris.",
            "Keep garden free of solanaceous weeds (e.g., horsenettle, nightshades)."
        ],
        "organic_treatment": [
            "Protective copper sprays applied every 7-10 days under rainy weather conditions."
        ],
        "prevention": "Adopt 3-year crop rotation and space plants at least 24-36 inches apart."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "title": "Two-Spotted Spider Mite (Tetranychus urticae)",
        "severity": "Medium-High",
        "symptoms": "Fine yellow stippling on leaf surface, bronzing, drying, and characteristic silky webbing on leaf undersides.",
        "cultural_control": [
            "Avoid drought stress; mites thrive in hot, dry, dusty environments.",
            "Hose down leaf undersides with strong water jets to disrupt webs and dislodge mite colonies."
        ],
        "organic_treatment": [
            "Apply insecticidal soap or horticultural neem oil spray directly to leaf undersides early in the morning.",
            "Introduce predatory mites (Phytoseiulus persimilis or Neoseiulus californicus)."
        ],
        "prevention": "Avoid broad-spectrum synthetic insecticides that kill beneficial predatory insects."
    },
    "Tomato__Target_Spot": {
        "title": "Target Spot in Tomato (Corynespora cassiicola)",
        "severity": "Medium",
        "symptoms": "Pinpoint brown spots enlarging into concentric rings with chlorotic yellow borders on leaves and sunken target-pattern fruit lesions.",
        "cultural_control": [
            "Improve field drainage and plant spacing.",
            "Promptly prune sucker branches to optimize sunlight penetration."
        ],
        "organic_treatment": [
            "Apply copper fungicides or bio-fungicide formulations containing Reynoutria sachalinensis extract."
        ],
        "prevention": "Avoid planting adjacent to older infected tomato or pepper blocks."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "title": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "severity": "Critical",
        "symptoms": "Severe upward leaf cupping, curling, mottled yellowing, stunted erect growth, and flower drop.",
        "cultural_control": [
            "TYLCV is vectored by whiteflies (Bemisia tabaci); management requires strict vector control.",
            "Use fine mesh insect netting (50-mesh) on greenhouse vents and nurseries.",
            "Rogue out and destroy infected plants immediately to prevent whitefly transmission."
        ],
        "organic_treatment": [
            "Deploy yellow sticky cards to monitor whitefly activity.",
            "Foliar neem oil or insecticidal soap applications targeted at immature whitefly nymphs."
        ],
        "prevention": "Plant TYLCV-resistant tomato cultivars (e.g., Tycoon, Securitas, Charger)."
    },
    "Tomato__Tomato_mosaic_virus": {
        "title": "Tomato Mosaic Virus (ToMV)",
        "severity": "High",
        "symptoms": "Light and dark green mosaic or mottled leaf patterns, distortion, 'fern-like' leaves, and internal browning of fruit.",
        "cultural_control": [
            "ToMV is mechanically transmitted via hands, tools, and tobacco products; practice strict handwashing with milk/soap.",
            "Dip pruning shears in 20% non-fat dry milk solution or 10% trisodium phosphate between plants.",
            "Immediately eliminate infected plants."
        ],
        "organic_treatment": [
            "No chemical cure exists for viral plant infections. Focus on hygiene and containment."
        ],
        "prevention": "Use certified virus-free seed treated with trisodium phosphate or dry heat."
    },
    "Tomato_healthy": {
        "title": "Healthy Tomato Leaf",
        "severity": "None",
        "symptoms": "Vigorous green foliage with well-defined leaflets and no lesion activity.",
        "cultural_control": [
            "Maintain consistent soil moisture to prevent blossom end rot and fruit splitting.",
            "Ensure regular fertilization balanced between nitrogen, phosphorus, and potassium."
        ],
        "organic_treatment": [
            "Apply prophylactic biological protectants (e.g., Bacillus subtilis) before prolonged overcast/damp weather."
        ],
        "prevention": "Regular scouting and crop rotation."
    }
}

WEATHER_RISK_ADVICE = {
    "Bacterial Leaf Blight": {
        "disease": "Bacterial Leaf Blight (Xanthomonas oryzae)",
        "risk_level": "High",
        "weather_driver": "Warm temperatures (25-34°C) with high humidity (>70%) and rainfall/strong wind gusts.",
        "preventive_actions": [
            "Avoid excessive nitrogen fertilizer application; split nitrogen doses into balanced, smaller applications.",
            "Ensure fields have proper water circulation and avoid deep flooding stagnant water.",
            "Disinfect paddy bunds and remove alternate weed hosts (Leersia hexandra, Cyperus species).",
            "If severe risk persists, consider preventive bactericide application (copper hydroxide or validamycin)."
        ]
    },
    "Brown Spot": {
        "disease": "Brown Spot (Bipolaris oryzae)",
        "risk_level": "Medium-High",
        "weather_driver": "Temperatures 25-30°C, high humidity, often aggravated by nutrient-deficient soils and water stress.",
        "preventive_actions": [
            "Correct soil nutritional deficiencies: ensure adequate Potassium (K), Silicon (Si), and organic matter.",
            "Avoid moisture stress or intermittent drying during tillering and panicle initiation.",
            "Soak seeds in warm water (52-54°C for 15 min) or treat with Trichoderma before sowing.",
            "Apply foliar potassium spray to strengthen leaf cell walls against fungal penetration."
        ]
    },
    "Healthy Rice Leaf": {
        "disease": "Favorable Conditions (Low Disease Risk)",
        "risk_level": "Low",
        "weather_driver": "Moderate temperatures and balanced humidity/rainfall that do not favor aggressive spore germination.",
        "preventive_actions": [
            "Maintain standard agronomic schedule and water management (alternate wetting and drying).",
            "Conduct routine bi-weekly field monitoring along field diagonals.",
            "Keep soil balanced with proper organic amendments."
        ]
    },
    "Leaf Blast": {
        "disease": "Rice Leaf Blast (Magnaporthe oryzae)",
        "risk_level": "Critical",
        "weather_driver": "Prolonged leaf wetness (>10 hours), high relative humidity (>90%), and overcast rain with cooler nights (18-24°C).",
        "preventive_actions": [
            "Cease any further top-dressing with nitrogen fertilizer until the risk subsides.",
            "Maintain shallow standing water (2-5 cm) rather than dry soil to reduce plant susceptibility.",
            "Ensure wide spacing (20 x 15 cm or wider) to allow wind movement and rapid drying.",
            "Prepare preventive bio-fungicides (Pseudomonas fluorescens) or recommended blast protectants (tricyclazole)."
        ]
    },
    "Leaf scald": {
        "disease": "Leaf Scald (Microdochium oryzae)",
        "risk_level": "Medium",
        "weather_driver": "Wet humid weather with frequent rain showers and moderate temperatures (22-28°C).",
        "preventive_actions": [
            "Avoid high seeding rates or close seedling transplant density.",
            "Balance fertilizer regime with adequate potassium and avoid excessive early nitrogen.",
            "Plow under rice stubble after previous harvests to reduce fungal inoculum."
        ]
    },
    "Sheath Blight": {
        "disease": "Sheath Blight (Rhizoctonia solani)",
        "risk_level": "High",
        "weather_driver": "High humidity (85-100%) and warm temperatures (28-32°C) inside dense plant canopies.",
        "preventive_actions": [
            "Drain excess water from paddies; lower water level to reduce humidity near plant bases.",
            "Eliminate dense canopy by judicious spacing and sensible nitrogen management.",
            "Remove floating sclerotia and organic scum during primary tillage and field preparation.",
            "Apply biocontrol agents like Trichoderma viride or validamycin if lesions advance to upper leaves."
        ]
    }
}

CROP_SOIL_GUIDANCE = {
    "rice": {
        "optimum_ph": "5.5 – 7.0",
        "water_req": "High (submerged / flooded conditions, 150-250 mm rainfall)",
        "npk_profile": "High Nitrogen, Moderate Phosphorus, Moderate Potassium",
        "soil_type": "Clayey or heavy loamy soils capable of holding water.",
        "tips": "Maintain continuous shallow standing water during tillering; drain 10 days before harvest."
    },
    "maize": {
        "optimum_ph": "5.8 – 7.0",
        "water_req": "Moderate (60-100 mm rainfall during growing season)",
        "npk_profile": "Heavy Nitrogen feeder, moderate Phosphorus & Potassium",
        "soil_type": "Well-drained deep loams rich in organic matter.",
        "tips": "Critical moisture periods are tasseling and silking; avoid waterlogging."
    },
    "cotton": {
        "optimum_ph": "6.0 – 8.0",
        "water_req": "Moderate-Low (50-80 mm rainfall, sunny dry picking season)",
        "npk_profile": "Balanced NPK; excess nitrogen causes vegetative lodging.",
        "soil_type": "Deep black cotton soils (vertisols) or alluvial loams.",
        "tips": "Requires warm growing season and completely dry weather during boll opening."
    },
    "coffee": {
        "optimum_ph": "5.0 – 6.5",
        "water_req": "High-Moderate (120-200 mm rainfall with good drainage)",
        "npk_profile": "High Potassium, moderate Nitrogen and Phosphorus",
        "soil_type": "Well-drained volcanic or loamy acidic soils on slopes.",
        "tips": "Thrives under partial shade in frost-free sub-tropical and highland climates."
    },
    "apple": {
        "optimum_ph": "5.5 – 6.5",
        "water_req": "Moderate (100-120 mm evenly distributed rainfall)",
        "npk_profile": "High Potassium for fruit sizing, balanced Nitrogen",
        "soil_type": "Deep, well-aerated loamy soils free of hardpan.",
        "tips": "Requires adequate winter chilling hours below 7°C for bud break."
    }
}


def get_disease_advice(disease_class: str) -> dict:
    """Return structured agronomic management advice for a plant disease class."""
    if disease_class in DISEASE_ADVICE:
        return DISEASE_ADVICE[disease_class]

    # Check if rice disease from WEATHER_RISK_ADVICE
    if disease_class in WEATHER_RISK_ADVICE:
        wr = WEATHER_RISK_ADVICE[disease_class]
        actions = wr.get("preventive_actions", [])
        return {
            "title": wr.get("disease", disease_class),
            "severity": wr.get("risk_level", "Medium"),
            "symptoms": wr.get("weather_driver", f"Symptoms characteristic of {disease_class}."),
            "cultural_control": actions[:2] if len(actions) >= 2 else actions,
            "organic_treatment": actions[2:] if len(actions) > 2 else ["Apply recommended organic or biological control agent."],
            "prevention": "Ensure balanced nutrition, clean certified seed sources, and regular paddy inspection."
        }

    # Generic fallback
    clean_name = disease_class.replace("__", " — ").replace("_", " ")
    return {
        "title": clean_name,
        "severity": "Moderate",
        "symptoms": f"Visible leaf abnormalities identified as {clean_name}.",
        "cultural_control": [
            "Isolate infected plants or prune visibly damaged foliage.",
            "Avoid overhead irrigation to minimize leaf surface wetness.",
            "Disinfect all cutting tools with a 70% alcohol or 10% bleach solution."
        ],
        "organic_treatment": [
            "Apply broad-spectrum preventive bio-fungicides (e.g. Bacillus subtilis or Neem oil).",
            "Ensure proper air circulation through the crop canopy."
        ],
        "prevention": "Rotate crops and use certified pathogen-free seeds."
    }


def get_weather_risk_advice(risk_class: str) -> dict:
    """Return preventive management advice for weather-induced disease risk."""
    if risk_class in WEATHER_RISK_ADVICE:
        return WEATHER_RISK_ADVICE[risk_class]

    return {
        "disease": risk_class,
        "risk_level": "Medium",
        "weather_driver": "Elevated humidity and rainfall conditions favor pathogen reproduction.",
        "preventive_actions": [
            "Inspect fields twice weekly for early lesion appearances.",
            "Avoid excess nitrogenous fertilizers which create soft, susceptible plant tissues.",
            "Ensure effective drainage channels to prevent stagnant surface water."
        ]
    }


def get_crop_guidance(crop_name: str, user_inputs: dict | None = None) -> dict:
    """Return agronomic profile and specific fertilizer feedback for the recommended crop."""
    crop_lower = crop_name.lower().strip()
    profile = CROP_SOIL_GUIDANCE.get(crop_lower, {
        "optimum_ph": "6.0 – 7.2",
        "water_req": "Moderate moisture with good drainage",
        "npk_profile": "Balanced N-P-K nutrition",
        "soil_type": "Loamy to sandy-loam fertile soil",
        "tips": "Implement regular weed management and maintain organic compost content."
    })

    feedback = []
    if user_inputs:
        ph = float(user_inputs.get("ph", 6.5))
        n = float(user_inputs.get("N", 50))
        p = float(user_inputs.get("P", 50))
        k = float(user_inputs.get("K", 50))
        rainfall = float(user_inputs.get("rainfall", 100))

        if ph < 5.5:
            feedback.append("Soil is acidic (pH < 5.5). Consider agricultural lime (calcium carbonate) application.")
        elif ph > 7.5:
            feedback.append("Soil is alkaline (pH > 7.5). Incorporate elemental sulfur or organic compost to lower pH.")

        if n < 30:
            feedback.append("Low Nitrogen (N). Supplement with compost, legume green manure, or urea.")
        elif n > 110:
            feedback.append("High Nitrogen (N). Refrain from heavy nitrogen top-dressing to prevent lodging and fungal vulnerability.")

        if k < 25:
            feedback.append("Low Potassium (K). Apply muriate of potash or wood ash to improve disease resistance and drought tolerance.")

        if rainfall < 40 and "rice" in crop_lower:
            feedback.append("Rainfall is low for rice; supplemental canal/tube-well irrigation will be essential.")

    return {
        "crop": crop_name,
        "profile": profile,
        "soil_adjustments": feedback if feedback else ["Soil N-P-K and pH levels are well within favorable thresholds."]
    }
