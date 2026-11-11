"""
app/services/chat_service.py

AI Agriculture Chatbot Service.
Integrates:
1. LLM API providers (Google Gemini, OpenAI / compatible)
2. Local offline Agricultural Knowledge Engine (fallback when no API key is provided)
3. Direct execution & explanation of existing ML models:
   - Crop Recommendation (predict_top3)
   - Plant Leaf Disease Detection (MobileNetV2 CNN via image upload)
   - Weather-Based Disease Risk (Support Vector Classifier / Environmental model)
   - Prediction History lookup from SQLite database
4. Multilingual support (English, Roman Urdu, Urdu).
"""

import os
import re
import json
import uuid
import requests
from datetime import datetime, timezone

import config
from app.models import Prediction
from app.app import db
from app.services.advice_service import (
    DISEASE_ADVICE,
    get_weather_risk_advice,
    get_crop_guidance
)

SYSTEM_PROMPT = """You are "AI Agriculture Assistant", an expert agronomist, crop specialist, and plant pathologist working for the AI Crop Advisor & Disease Detector system.
Your mission is to provide helpful, practical, farmer-friendly agricultural advice.

Key Guidelines:
1. Expertise: Soil nutrients (Nitrogen, Phosphorus, Potassium - NPK), soil pH, humidity, temperature, rainfall, crop cultivation, plant pathology, and sustainable farming.
2. Safety & Responsibility: Always clarify that advice is an academic/decision-support recommendation, not a certified agronomic diagnosis. Advise consulting local agricultural extension officers for large-scale interventions. Prioritize integrated pest management (IPM), cultural sanitation, and organic controls over toxic chemical misuse.
3. Language: Respond in the user's preferred language. If the user asks in Roman Urdu (e.g., "meri matti ka pH 6 hai", "fasal konsi lagayein"), respond in clear, friendly Roman Urdu. If in English, respond in English. If in Urdu script, respond in Urdu script.
4. Accuracy: Explain confidence scores and model predictions honestly. If specific data is missing to make a prediction, politely request the necessary parameters.
5. Tone: Polite, clear, encouraging, and farmer-friendly.
"""

# ─── Reference crop profiles for local knowledge engine ───────────────────────
CROP_PROFILES = {
    "rice": {"n": "60-100", "p": "35-60", "k": "35-50", "temp": "20-27°C", "humidity": "80-90%", "ph": "5.5-7.0", "rainfall": "180-300mm", "season": "Kharif / Monsoon"},
    "maize": {"n": "60-100", "p": "40-60", "k": "15-30", "temp": "18-27°C", "humidity": "55-75%", "ph": "5.8-7.2", "rainfall": "60-110mm", "season": "Kharif / Spring"},
    "cotton": {"n": "100-140", "p": "35-60", "k": "15-30", "temp": "22-30°C", "humidity": "65-85%", "ph": "6.0-7.8", "rainfall": "60-100mm", "season": "Kharif"},
    "wheat": {"n": "80-120", "p": "40-60", "k": "20-40", "temp": "15-22°C", "humidity": "50-70%", "ph": "6.0-7.5", "rainfall": "50-100mm", "season": "Rabi / Winter"},
    "chickpea": {"n": "20-40", "p": "55-80", "k": "75-90", "temp": "15-25°C", "humidity": "15-25%", "ph": "6.0-7.5", "rainfall": "60-90mm", "season": "Rabi"},
    "kidneybeans": {"n": "15-35", "p": "55-80", "k": "15-30", "temp": "15-24°C", "humidity": "50-65%", "ph": "5.5-6.5", "rainfall": "60-150mm", "season": "Kharif/Rabi"},
    "apple": {"n": "15-35", "p": "120-145", "k": "190-205", "temp": "21-24°C", "humidity": "90-95%", "ph": "5.5-6.5", "rainfall": "100-125mm", "season": "Perennial temperate"},
    "banana": {"n": "90-120", "p": "70-95", "k": "45-55", "temp": "25-30°C", "humidity": "75-85%", "ph": "5.5-6.8", "rainfall": "90-120mm", "season": "Year-round tropical"},
    "mango": {"n": "10-30", "p": "15-35", "k": "25-35", "temp": "27-36°C", "humidity": "45-55%", "ph": "5.0-7.0", "rainfall": "90-100mm", "season": "Summer fruit"},
    "coffee": {"n": "80-120", "p": "15-40", "k": "25-35", "temp": "23-28°C", "humidity": "55-70%", "ph": "6.0-7.0", "rainfall": "115-200mm", "season": "Tropical highland"},
    "jute": {"n": "60-90", "p": "35-50", "k": "35-45", "temp": "23-26°C", "humidity": "70-90%", "ph": "6.0-7.5", "rainfall": "150-200mm", "season": "Monsoon"},
}


def detect_roman_urdu(text: str) -> bool:
    """Detect if the message is in Roman Urdu or Urdu."""
    roman_urdu_words = {
        "meri", "mera", "mere", "fasal", "khet", "mitti", "zameen", "pani",
        "chawal", "bimari", "kharab", "karo", "kia", "kya", "konsi", "kon",
        "kaise", "ilaj", "patay", "pata", "keeray", "khad", "peela", "sookh",
        "daag", "dhabay", "barish", "hawa", "batao", "bataien", "hai", "hain",
        "shukriya", "madad", "gandum", "makai", "kapas"
    }
    # Check Urdu Unicode range
    if any("\u0600" <= c <= "\u06FF" for c in text):
        return True
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    matches = tokens.intersection(roman_urdu_words)
    return len(matches) >= 2 or (len(matches) == 1 and len(tokens) <= 4)


def extract_crop_params(text: str) -> dict | None:
    """Extract N, P, K, temp, humidity, pH, rainfall from text if present."""
    patterns = {
        "N": [r"\bn(?:itrogen)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", r"([0-9]+(?:\.[0-9]+)?)\s*(?:ppm|kg/ha)?\s*nitrogen"],
        "P": [r"\bp(?:hosphorus)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", r"([0-9]+(?:\.[0-9]+)?)\s*phosphorus"],
        "K": [r"\bk(?:potassium)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", r"([0-9]+(?:\.[0-9]+)?)\s*potassium"],
        "temperature": [r"\btemp(?:erature)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", r"([0-9]+(?:\.[0-9]+)?)\s*(?:°c|celsius|degrees)"],
        "humidity": [r"\bhumid(?:ity)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", r"([0-9]+(?:\.[0-9]+)?)\s*%?\s*humidity"],
        "ph": [r"\bph\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", r"soil\s*ph\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)"],
        "rainfall": [r"\brain(?:fall)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", r"([0-9]+(?:\.[0-9]+)?)\s*mm\s*rain"],
    }
    extracted = {}
    lower = text.lower()
    for param, regexes in patterns.items():
        for r in regexes:
            m = re.search(r, lower)
            if m:
                try:
                    extracted[param] = float(m.group(1))
                    break
                except ValueError:
                    continue

    # Return if at least 4 params are present
    if len(extracted) >= 4:
        return extracted
    return None


def extract_weather_params(text: str) -> dict | None:
    """Extract temperature, humidity, rainfall, wind speed for weather disease risk."""
    lower = text.lower()
    extracted = {}
    m_temp = re.search(r"\btemp(?:erature)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", lower)
    m_hum = re.search(r"\bhumid(?:ity)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", lower)
    m_rain = re.search(r"\brain(?:fall)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", lower)
    m_wind = re.search(r"\bwind(?:_speed)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", lower)

    if m_temp: extracted["temperature"] = float(m_temp.group(1))
    if m_hum:  extracted["humidity"] = float(m_hum.group(1))
    if m_rain: extracted["rainfall"] = float(m_rain.group(1))
    if m_wind: extracted["wind_speed"] = float(m_wind.group(1))

    if len(extracted) >= 3:
        return extracted
    return None


def run_crop_model_prediction(params: dict) -> dict:
    """Run genuine trained crop model."""
    from src.crop.predict import predict_top3
    # Use defaults for missing params if partial
    defaults = {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 25.0, "humidity": 70.0, "ph": 6.5, "rainfall": 100.0}
    full_params = {k: params.get(k, defaults[k]) for k in defaults}
    res = predict_top3(**full_params)
    return {"inputs": full_params, "predictions": res}


def run_weather_risk_prediction(params: dict) -> dict:
    """Run genuine trained weather risk model."""
    from src.weather_disease.predict import predict
    defaults = {"temperature": 25.0, "humidity": 85.0, "rainfall": 10.0, "wind_speed": 15.0}
    full_params = {k: params.get(k, defaults[k]) for k in defaults}
    res = predict(**full_params)
    return {"inputs": full_params, "result": res}


def run_image_disease_prediction(image_path: str, model_type: str = "garden") -> dict:
    """Run genuine trained MobileNetV2 disease model on uploaded image."""
    model_type = "rice" if model_type == "rice" else "garden"
    if model_type == "rice":
        from src.disease.predict_rice import predict_rice_from_path
        pred = predict_rice_from_path(image_path)
    else:
        from src.disease.predict import predict_from_path
        pred = predict_from_path(image_path)

    disease_name = pred.get("predicted_class") or pred.get("disease", "Unknown")
    confidence = float(pred.get("confidence", 0.0))
    is_low = pred.get("low_confidence", pred.get("is_low_confidence", confidence < 0.60))

    # Fetch advice
    advice = DISEASE_ADVICE.get(disease_name, None)

    # Log in SQLite database
    try:
        entry = Prediction(
            prediction_type="disease",
            input_data=json.dumps({"filename": os.path.basename(image_path), "model_type": model_type}),
            disease_prediction=disease_name,
            confidence=confidence
        )
        db.session.add(entry)
        db.session.commit()
    except Exception:
        db.session.rollback()

    return {
        "model_type": model_type,
        "disease": disease_name,
        "confidence": confidence,
        "is_low_confidence": is_low,
        "top3": pred.get("top3", []),
        "advice": advice
    }


def get_latest_prediction_summary() -> str | None:
    """Query recent prediction history from SQLite DB."""
    try:
        latest = Prediction.query.order_by(Prediction.created_at.desc()).first()
        if not latest:
            return None
        dt_str = latest.timestamp.strftime("%b %d, %Y %I:%M %p") if latest.timestamp else "Recently"
        conf_str = f" with {latest.confidence * 100:.1f}% confidence" if latest.confidence is not None else ""
        if latest.prediction_type == "crop":
            return f"Most recent prediction was **Crop Recommendation** on {dt_str}: Recommended **{latest.crop_prediction}**{conf_str}."
        elif latest.prediction_type == "disease":
            return f"Most recent prediction was **Disease Detection** on {dt_str}: Detected **{latest.disease_prediction}**{conf_str}."
        elif latest.prediction_type == "weather":
            return f"Most recent prediction was **Weather Risk Assessment** on {dt_str}: Predicted risk **{latest.risk_prediction}**{conf_str}."
        return f"Recent {latest.prediction_type} prediction recorded{conf_str}."
    except Exception:
        return None


# ─── LLM API Callers ──────────────────────────────────────────────────────────

def call_gemini_api(user_message: str, history: list) -> str:
    """Call Google Gemini 1.5/2.0 API via REST."""
    api_key = config.GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured in .env")

    model_name = config.CHATBOT_MODEL or "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # Build contents array
    contents = []
    # Add recent history (last 6 messages max)
    for msg in history[-6:]:
        role = "user" if msg.get("role") == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg.get("content", "")}]
        })
    contents.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 900,
        }
    }

    resp = requests.post(url, json=payload, timeout=15)
    if resp.status_code != 200:
        err_msg = resp.text
        try:
            err_json = resp.json()
            err_msg = err_json.get("error", {}).get("message", resp.text)
        except Exception:
            pass
        raise RuntimeError(f"Gemini API returned status {resp.status_code}: {err_msg}")

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("No response candidate returned by Gemini API")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


def call_openai_api(user_message: str, history: list) -> str:
    """Call OpenAI or compatible API via REST."""
    api_key = config.OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured in .env")

    url = f"{config.OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history[-6:]:
        role = "user" if msg.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": msg.get("content", "")})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": config.OPENAI_MODEL or "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 800,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API status {resp.status_code}: {resp.text}")

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("No completion returned by OpenAI API")
    return choices[0].get("message", {}).get("content", "").strip()


# ─── Local Agricultural Knowledge Engine (Offline Fallback) ───────────────────

def local_agriculture_response(user_message: str, is_roman_urdu: bool = False) -> str:
    """
    Intelligent rule-and-knowledge agricultural response engine.
    Used when no external LLM API key is configured.
    Honest, comprehensive, and scientifically accurate.
    """
    lower = user_message.lower()

    # 1. Soil Fertility & NPK
    if any(k in lower for k in ["npk", "nitrogen", "phosphorus", "potassium", "fertility", "khad", "fertilizer"]):
        if is_roman_urdu:
            return (
                "🌱 **Mitti ki Zarkhaizi (Soil Fertility) aur NPK ki Rahnumai:**\n\n"
                "• **Nitrogen (N):** Paton ki sabz rangat aur nashonuma k liye zaroori hai. Kami ki soorat me patay peelay par jatay hain (Urea ya compost istemal karein).\n"
                "• **Phosphorus (P):** Jaron ki mazbooti aur phool/beej banne me madad deta hai (DAP ya single superphosphate faidamand hai).\n"
                "• **Potassium (K):** Fasal ko beemariyon aur sookhay (drought) se bachata hai aur paidawar ka wazan barhata hai (SOP/MOP).\n\n"
                "💡 *Mashwara:* Hamesha fasal boye jane se pehle apni mitti ka lab test karwayen taake munasib miqdar me khad dali ja sakay."
            )
        return (
            "🌱 **Soil Fertility & NPK Management Guide:**\n\n"
            "• **Nitrogen (N):** Vital for vegetative growth and chlorophyll synthesis. Deficiency causes leaf chlorosis (yellowing) starting from older lower leaves.\n"
            "• **Phosphorus (P):** Critical for deep root establishment, early tillering, and uniform flowering.\n"
            "• **Potassium (K):** Regulates stomatal conductance, builds stem resistance, and activates disease defense enzymes.\n\n"
            "💡 *Best Practice:* Combine organic compost/farmyard manure with targeted mineral fertilizers based on periodic soil testing."
        )

    # 2. Soil pH
    if any(k in lower for k in ["soil ph", "ph level", "acidic", "alkaline", "mitti ka ph"]):
        if is_roman_urdu:
            return (
                "🧪 **Mitti k pH ki Maloomat:**\n\n"
                "• **Ideal pH Range:** Ziada tar faslon k liye **6.0 se 7.2** behtareen range hai.\n"
                "• **Tezabi Mitti (Acidic, pH < 6.0):** Agar mitti acidic ho to agricultural lime (choona / CaCO₃) shamil karein.\n"
                "• **Khari / Shora Zameen (Alkaline, pH > 7.5):** Agar pH ziada ho to Gypsum (gach) ya organic matter (gobar ki khad) dalen taake nutrients asani se jazb hon."
            )
        return (
            "🧪 **Soil pH & Nutrient Availability:**\n\n"
            "• **Optimal Range:** Most field crops thrive in a neutral-to-slightly-acidic range of **6.0 – 7.2**.\n"
            "• **Acidic Soils (pH < 6.0):** Treat with agricultural limestone (calcium carbonate) to neutralize excess acidity and unlock phosphorus.\n"
            "• **Alkaline / Calcareous Soils (pH > 7.8):** Apply agricultural gypsum ($CaSO_4$) along with elemental sulfur or heavy organic compost to reduce sodicity and improve iron/zinc uptake."
        )

    # 3. Leaf Blast in Rice
    if any(k in lower for k in ["leaf blast", "blast disease", "blast ka ilaj", "blast bimari"]):
        advice = DISEASE_ADVICE.get("Leaf Blast", {})
        if is_roman_urdu:
            return (
                "🌾 **Chawal ka Leaf Blast (Magnaporthe oryzae):**\n\n"
                "• **Waja (Causes):** 20°C–28°C darja hararat, 90% se ziada nami (humidity), aur hawa me spores ka phailao.\n"
                "• **Alamaat (Symptoms):** Paton par spindle-shaped (chakku jese) dhabay bante hain jin k darmian me grey rang aur kinare brown hotay hain.\n"
                "• **Ilaj aur Hifazat:**\n"
                "  1. Khet me se zayed pani nikal kar aeration behtar karein.\n"
                "  2. Nitrogen khad ki boht ziada miqdar se gurez karein kyun k ye bimari ko barhati hai.\n"
                "  3. Shadeed hamlay me Tricyclazole ya Isoprothiolane ka munasib spray karein."
            )
        return (
            "🌾 **Rice Leaf Blast (*Magnaporthe oryzae*):**\n\n"
            "• **Triggers:** Prolonged leaf wetness, humidity >90%, temperatures between 20–28°C, and excessive Nitrogen fertilization.\n"
            "• **Symptoms:** Diamond/spindle-shaped lesions with gray-white centers and reddish-brown borders.\n"
            "• **Management:**\n"
            "  1. Drain excess standing water periodically to facilitate field aeration.\n"
            "  2. Split Nitrogen doses rather than applying heavy single top-dressings.\n"
            "  3. Use certified blast-resistant seed varieties; apply biocontrol *Bacillus subtilis* or protective fungicides when risk indicators spike."
        )

    # 4. Brown Spot in Rice
    if any(k in lower for k in ["brown spot", "brown dhabay", "brown spots"]):
        if is_roman_urdu:
            return (
                "🌾 **Chawal ka Brown Spot (Bipolaris oryzae):**\n\n"
                "• **Waja:** Ye beemari aam tor par un zameeno me aati hai jahan Potassium (K) ya Manganese ki kami ho ya paani ka tanao ho.\n"
                "• **Alamaat:** Paton par gol aur oval shakal k brown rang k dhabay jin k ird gird peela halqa (yellow halo) hota hai.\n"
                "• **Hal:** Potash khad (SOP) ka munasib istemal karein aur zameen ko sokha na lagne dein."
            )
        return (
            "🌾 **Rice Brown Spot (*Bipolaris oryzae*):**\n\n"
            "• **Triggers:** Highly correlated with nutrient stress—specifically low Potassium ($K$), low soil Silicon, and drought-stressed soils.\n"
            "• **Symptoms:** Small, circular to oval sesame-seed-like dark brown spots with light-colored centers and yellow halos across leaf blades.\n"
            "• **Management:** Correct soil potassium deficiencies, maintain steady field saturation, and avoid drought-induced root stress."
        )

    # 5. Humidity & Weather Impact on Plant Diseases
    if any(k in lower for k in ["humidity", "weather risk", "nami", "hawa", "rainfall", "barish"]):
        if is_roman_urdu:
            return (
                "🌦️ **Mausam aur Beemariyon ka Taluq (Weather Disease Risk):**\n\n"
                "• **Ziada Nami (Humidity > 80%):** Fungal aur bacterial jaraseem k phailne k liye sab se mufeed mahol hai.\n"
                "• **Barish (Rainfall):** Barish k qatre jaraseem ko aik patay se doosre patay par splash karte hain.\n"
                "• **Tez Hawa (Wind):** Fungal spores ko kafi door tak urra kar le jati hai.\n\n"
                "💡 *Ahtiyati Tadbeer:* Hamare **Weather Risk** module me ja kar apne ilaqe ka mausam daalen aur pehle se ahtiyat karein."
            )
        return (
            "🌦️ **How Weather Drives Crop Disease Outbreaks:**\n\n"
            "• **Relative Humidity (>80%):** Fungal spores require free moisture or micro-humidity envelopes on leaf surfaces to germinate and penetrate plant stomata.\n"
            "• **Rainfall:** Rain droplets cause pathogen splash-dispersal, washing inoculants down stem sheaths.\n"
            "• **Wind Speed:** Strong winds disperse microscopic airborne conidia across entire adjoining acreage.\n\n"
            "💡 *Proactive Strategy:* Use our integrated **Weather Risk** tab to simulate upcoming forecasts and schedule proactive protective measures."
        )

    # 6. General Crop Recommendation Inquiry
    if any(k in lower for k in ["which crop", "suitable crop", "konsi fasal", "what to grow", "crop recommendation"]):
        if is_roman_urdu:
            return (
                "🌾 **Fasal ki Tajweez (Crop Recommendation):**\n\n"
                "Sahi fasal ka intikhab karne k liye humari AI model 7 cheezon ko dekhti hai:\n"
                "1. **Nitrogen (N)**, **Phosphorus (P)**, aur **Potassium (K)** ki miqdar\n"
                "2. **Darja Hararat (Temperature)** aur **Nami (Humidity)**\n"
                "3. **Mitti ka pH** aur **Salana Barish (Rainfall)**\n\n"
                "👉 Aap yahan chat me apne parameters likh sakte hain (maslan: `N=90, P=42, K=43, temp=24, humidity=80, ph=6.5, rain=150`), ya ooper diye gaye **Crop Advisor** page par ja kar live test kar sakte hain!"
            )
        return (
            "🌾 **Crop Recommendation System:**\n\n"
            "Our tuned Random Forest model evaluates 7 soil-climate parameters to predict the top-3 optimal crops:\n"
            "• Soil N-P-K nutrient concentrations\n"
            "• Ambient Temperature & Relative Humidity\n"
            "• Soil pH & Mean Annual Rainfall\n\n"
            "👉 **Try it here in chat:** Type your values like `N=90, P=42, K=43, temp=22, humidity=80, ph=6.5, rainfall=200` or navigate to the **Crop Advisor** tab for interactive sliders and SHAP explainability!"
        )

    # 7. Greeting & Default Help
    if any(k in lower for k in ["hello", "hi", "salam", "assalam", "hey", "help", "madad"]):
        if is_roman_urdu:
            return (
                "Assalam-o-Alaikum! Main aapka **AI Agriculture Assistant** hoon. 🌾\n\n"
                "Aap mujh se ye cheezain pooch sakte hain:\n"
                "• Apni mitti k mutabiq behtareen fasal ki tajweez.\n"
                "• Paton ki beemariyon ki tashkhees aur organic ilaj.\n"
                "• Mitti k NPK aur pH ko behtar banane k tareeqay.\n"
                "• Chawal k khet me bimariyon (Leaf Blast, Brown Spot) se hifazat.\n"
                "• Hamare AI models k nataij aur scores ki wazahat.\n\n"
                "Bataiye, aaj main aapki kya madad kar sakta hoon?"
            )
        return (
            "Hello! I am your **AI Agriculture Assistant**. 🌾\n\n"
            "I can help you with:\n"
            "• **Crop Suitability:** Tell me your soil N-P-K, pH, and weather conditions for instant recommendations.\n"
            "• **Plant Disease Diagnosis:** Ask about leaf symptoms or upload a leaf photo directly in this chat.\n"
            "• **Weather Risk:** Check if high humidity or rainfall threatens your fields.\n"
            "• **Soil Health:** Tips to improve fertility and balance pH.\n\n"
            "How can I assist your farm today?"
        )

    # 8. Fallback general agricultural response
    if is_roman_urdu:
        return (
            "Ziraat (Agriculture) k hawale se aapka sawal mosool hua hai. 🌾\n\n"
            "Behtar rehnumai k liye aap mitti k ajza (NPK), fasal ka naam, ya beemari ki alamaat (jaise peelay patay, brown spots, sookha) wazeh taur par likhein.\n\n"
            "💡 *Tip:* Agar aapko fasal ki recommendation chahiye to N, P, K, pH aur barish ki values share karein."
        )

    return (
        "I understand you are asking about agriculture and crop management. 🌾\n\n"
        "To give you the most accurate agronomic advice, could you share a bit more detail?\n"
        "• If asking for a **crop recommendation**, please share your soil N-P-K, pH, and rainfall.\n"
        "• If asking about a **disease**, describe the leaf symptoms (e.g. brown spots, yellow margins, powder) or upload an image using the attachment button below.\n"
        "• You can also consult the **Crop Advisor** or **Disease Detection** tabs for complete automated analysis."
    )


# ─── Master Chat Handler ──────────────────────────────────────────────────────

def process_chat_message(
    user_message: str,
    history: list | None = None,
    uploaded_image_path: str | None = None,
    model_type: str = "garden"
) -> dict:
    """
    Main entry point for chatbot query processing.
    Validates input, routes to models, calls LLM or local engine.
    """
    if history is None:
        history = []

    user_message = (user_message or "").strip()
    if not user_message and not uploaded_image_path:
        return {"success": False, "error": "Message or image is required."}

    # Length safeguard
    if len(user_message) > 2000:
        return {"success": False, "error": "Message exceeds maximum allowed length of 2,000 characters."}

    is_roman_urdu = detect_roman_urdu(user_message)

    # ── Case A: Leaf Image Uploaded inside Chat ────────────────────────────────
    if uploaded_image_path and os.path.exists(uploaded_image_path):
        try:
            diag = run_image_disease_prediction(uploaded_image_path, model_type=model_type)
            dis_name = diag["disease"]
            conf = diag["confidence"] * 100
            advice = diag["advice"]

            reply = f"🔬 **AI Plant Disease Diagnosis ({diag['model_type'].capitalize()} Model):**\n\n"
            reply += f"• **Identified Condition:** **{dis_name.replace('___', ' - ').replace('__', ' ')}**\n"
            reply += f"• **Prediction Confidence:** **{conf:.1f}%**\n"

            if diag["is_low_confidence"]:
                reply += "⚠️ *Notice: Low confidence (<60%). Please verify the leaf image in clear daylight to ensure accurate diagnosis.*\n\n"

            if advice:
                reply += f"• **Severity:** {advice.get('severity', 'Medium')}\n"
                reply += f"• **Symptoms:** {advice.get('symptoms', 'None reported.')}\n\n"
                reply += "🌿 **Recommended Cultural Controls:**\n"
                for step in advice.get("cultural_control", []):
                    reply += f"  - {step}\n"
                reply += "\n🛡️ **Organic / Preventive Treatments:**\n"
                for treat in advice.get("organic_treatment", []):
                    reply += f"  - {treat}\n"
            else:
                reply += "\n*Guidance: Remove infected foliage, avoid overhead watering, and ensure adequate spacing.*"

            reply += "\n\n📋 *Disclaimer: AI prediction is a decision-support indicator. Consult your local agricultural extension for critical crop interventions.*"

            return {
                "success": True,
                "reply": reply,
                "provider": "local_vision_model",
                "is_model_prediction": True,
                "prediction_type": "disease",
                "prediction_data": diag
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Image diagnosis failed: {str(e)}"
            }

    # ── Case B: User asks for previous history ─────────────────────────────────
    if any(q in user_message.lower() for q in ["last prediction", "previous prediction", "my history", "pichli prediction", "record dikhao"]):
        summary = get_latest_prediction_summary()
        if summary:
            return {
                "success": True,
                "reply": f"📜 **Database Record Lookup:**\n\n{summary}\n\nYou can view full past entries anytime on the **[History](/history)** page.",
                "provider": "database",
                "is_model_prediction": False
            }
        else:
            return {
                "success": True,
                "reply": "📜 No previous predictions found in the database yet. Run a test in the Crop Advisor, Disease Detection, or Weather Risk pages to record history entries.",
                "provider": "database",
                "is_model_prediction": False
            }

    # ── Case C: User provided Crop Parameters in text ─────────────────────────
    crop_params = extract_crop_params(user_message)
    if crop_params:
        try:
            pred = run_crop_model_prediction(crop_params)
            top = pred["predictions"][0]
            sec = pred["predictions"][1] if len(pred["predictions"]) > 1 else None
            thd = pred["predictions"][2] if len(pred["predictions"]) > 2 else None

            p_in = pred["inputs"]
            reply = "🌾 **Trained Crop Recommendation Model Output:**\n\n"
            reply += f"Based on your soil & climate parameters (N={p_in['N']}, P={p_in['P']}, K={p_in['K']}, pH={p_in['ph']}, Rain={p_in['rainfall']}mm, Temp={p_in['temperature']}°C):\n\n"
            reply += f"1. 🥇 **{top['crop'].capitalize()}** — **{top['probability'] * 100:.1f}% Confidence**\n"
            if sec:
                reply += f"2. 🥈 **{sec['crop'].capitalize()}** — {sec['probability'] * 100:.1f}%\n"
            if thd:
                reply += f"3. 🥉 **{thd['crop'].capitalize()}** — {thd['probability'] * 100:.1f}%\n"

            # Log to DB
            try:
                db.session.add(Prediction(
                    prediction_type="crop",
                    input_data=json.dumps(p_in),
                    crop_prediction=top["crop"],
                    confidence=top["probability"]
                ))
                db.session.commit()
            except Exception:
                db.session.rollback()

            reply += "\n💡 *Agronomic Note:* For full SHAP explainability feature impact and What-If simulations, visit the **[Crop Advisor](/crop-advisor)** page."
            return {
                "success": True,
                "reply": reply,
                "provider": "local_crop_model",
                "is_model_prediction": True,
                "prediction_type": "crop",
                "prediction_data": pred
            }
        except Exception as e:
            pass  # Fall through to standard conversational pipeline

    # ── Case D: User provided Weather Disease Risk Parameters in text ─────────
    weather_params = extract_weather_params(user_message)
    if weather_params and any(w in user_message.lower() for w in ["disease", "risk", "khatra", "blast", "weather"]):
        try:
            w_pred = run_weather_risk_prediction(weather_params)
            res = w_pred["result"]
            r_in = w_pred["inputs"]
            reply = "🌦️ **Trained Weather Disease Risk Model Output:**\n\n"
            reply += f"Environmental conditions analyzed: Temp={r_in['temperature']}°C, Humidity={r_in['humidity']}%, Rain={r_in['rainfall']}mm, Wind={r_in['wind_speed']}km/h\n\n"
            reply += f"• **Vulnerability State:** **{res['predicted_disease']}**\n"
            reply += f"• **Risk Probability:** **{res['probability'] * 100:.1f}%**\n\n"

            adv = get_weather_risk_advice(res["predicted_disease"])
            if adv:
                reply += f"🛡️ **Proactive Guidance:** {adv.get('action', '')}\n"

            reply += "\n⚠️ *Note: Weather-based statistical risk prediction for rice crops only.*"

            # Log to DB
            try:
                db.session.add(Prediction(
                    prediction_type="weather",
                    input_data=json.dumps(r_in),
                    risk_prediction=res["predicted_disease"],
                    confidence=res["probability"]
                ))
                db.session.commit()
            except Exception:
                db.session.rollback()

            return {
                "success": True,
                "reply": reply,
                "provider": "local_weather_model",
                "is_model_prediction": True,
                "prediction_type": "weather",
                "prediction_data": w_pred
            }
        except Exception:
            pass

    # ── Case E: Conversational LLM API or Local Fallback ───────────────────────
    provider = config.CHATBOT_PROVIDER  # 'auto', 'gemini', 'openai', 'local'
    llm_error = None

    if provider in ["auto", "gemini"] and config.GEMINI_API_KEY:
        try:
            ai_reply = call_gemini_api(user_message, history)
            return {
                "success": True,
                "reply": ai_reply,
                "provider": "gemini",
                "is_model_prediction": False
            }
        except Exception as e:
            llm_error = f"Gemini API Error: {str(e)}"

    if (provider in ["auto", "openai"] or (provider == "gemini" and llm_error)) and config.OPENAI_API_KEY:
        try:
            ai_reply = call_openai_api(user_message, history)
            return {
                "success": True,
                "reply": ai_reply,
                "provider": "openai",
                "is_model_prediction": False
            }
        except Exception as e:
            llm_error = f"OpenAI API Error: {str(e)}"

    # ── Fallback: Local Agricultural Expert Engine ────────────────────────────
    # Transparently inform user about the local engine status
    local_reply = local_agriculture_response(user_message, is_roman_urdu=is_roman_urdu)

    note = ""
    if not config.GEMINI_API_KEY and not config.OPENAI_API_KEY:
        # Honest setup note
        note = "\n\n*(🌱 Operating via Built-in Agriculture Expert Engine. To enable generative Google Gemini LLM, add `GEMINI_API_KEY` to your `.env` file.)*"
    elif llm_error:
        note = f"\n\n*(⚠️ Note: External AI provider temporarily unreachable [{llm_error}]. Showing verified local agricultural guidance above.)*"

    return {
        "success": True,
        "reply": local_reply + note,
        "provider": "local_expert_engine",
        "is_model_prediction": False,
        "llm_notice": note.strip()
    }
