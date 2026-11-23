import streamlit as st
from PIL import Image
from google import genai
from transformers import pipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DermaScan AI", page_icon="🩺", layout="centered")

# --- 1. LOAD THE SPECIALIST MODEL (Cached for Speed) ---
@st.cache_resource
def load_medical_model():
    """
    Loads the Hugging Face model once and keeps it in memory.
    This model is trained to classify 7 types of skin lesions (HAM10000).
    """
    # Using a standard model fine-tuned on skin cancer datasets
    # You can swap this string for other models from Hugging Face
    return pipeline("image-classification", model="Anwarkh1/Skin_Cancer-Image_Classification")

# Load model immediately so it's ready
with st.spinner("Loading medical diagnostic models..."):
    classifier = load_medical_model()

# --- 2. UI LAYOUT ---
st.title("🩺 DermaScan AI Assistant")
st.markdown("""
This tool uses a **Hybrid AI Architecture**:
1. **Computer Vision** analyzes the lesion pixels.
2. **Gemini** explains the findings.
*Disclaimer: For educational purposes only. Not a medical device.*
""")

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    st.info("Get your key at [aistudio.google.com](https://aistudio.google.com/)")

# --- 3. MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Upload a clear photo of the lesion", type=["jpg", "png", "jpeg"])

if uploaded_file and api_key:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Scan", use_container_width=True)

    if st.button("Analyze Lesion"):
        with st.spinner("Analyzing cell patterns..."):
            # A. THE SPECIALIST (Vision)
            # The pipeline expects a path or PIL image
            predictions = classifier(image)
            top_result = predictions[0]
            diagnosis = top_result['label']
            confidence = top_result['score']

            # Display raw data for transparency
            st.subheader("Technical Analysis")
            # Create a progress bar for confidence
            st.progress(confidence)
            st.write(f"**Detected Condition:** {diagnosis}")
            st.write(f"**Confidence Score:** {confidence:.2%}")

            # B. THE COMMUNICATOR (Gemini)
            client = genai.Client(api_key=api_key)

            prompt = f"""
            You are a specialized dermatologist assistant. 
            A computer vision model has analyzed a patient's skin lesion with the following result:
            - Condition: {diagnosis}
            - Confidence: {confidence:.2%}
            
            Please provide a response for the user that:
            1. Explains what '{diagnosis}' is in simple, non-alarmist language.
            2. Suggests specific next steps (e.g., "See a doctor immediately" vs "Monitor changes").
            3. Uses a calm, professional tone.
            4. STRICTLY ENDS with a disclaimer that you are an AI and this is not a medical diagnosis.
            """
            
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt,
                )
                response_placeholder.write(response.text)

elif not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar to proceed.")