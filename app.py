import streamlit as st
import numpy as np
from PIL import Image
import random
import time
# Add these imports for the real model
from transformers import ViTImageProcessor, TFViTForImageClassification
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="Handwriting Personality Analysis",
    page_icon="🖋️",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained ViT model (REAL VERSION)"""
    try:
        with st.spinner("Loading AI model... This may take a moment."):
            MODEL_NAME = "google/vit-base-patch16-224"
            processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
            model = TFViTForImageClassification.from_pretrained("../handwritten-personality-vision_transformer copy/personality_vit_20250521-144334")
            return model, processor
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.error("Currently using dummy mode fallback")
        time.sleep(1)
        return "dummy_model", "dummy_processor"

def preprocess_image(image, processor):
    """Preprocess image for ViT model (REAL VERSION)"""
    if processor == "dummy_processor":
        # Dummy mode
        return np.array(image)
    else:
        # Real model preprocessing
        try:
            inputs = processor(images=image, return_tensors="tf")
            return inputs
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None

def predict_personality(processed_inputs, model):
    """
    Real prediction function using the trained ViT model
    Returns: dictionary with personality traits and their percentages
    """
    if model == "dummy_model":
        # Dummy mode FALLBACK ONLY. HAPUS SAAT RILIS
        time.sleep(2)
        traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        raw_values = [random.uniform(0.1, 0.4) for _ in traits]
        total = sum(raw_values)
        percentages = [(val / total) for val in raw_values]
        results = {}
        for trait, percentage in zip(traits, percentages):
            results[trait] = percentage
        return results
    
    else:
        # ViT model(?) logic
        try:
            predictions = model(**processed_inputs)
              # Convert logits to probabilities using softmax
            probabilities = tf.nn.softmax(predictions.logits, axis=-1).numpy()
            
            # OCEAN traits mapping based on actual model's id2label mapping
            # Order from model config.json: 0:Agreeableness, 1:Conscientiousness, 2:Extraversion, 3:Neuroticism, 4:Openness
            ocean_traits = {
                'Agreeableness': float(probabilities[0][0]),
                'Conscientiousness': float(probabilities[0][1]),
                'Extraversion': float(probabilities[0][2]),
                'Neuroticism': float(probabilities[0][3]),
                'Openness': float(probabilities[0][4])
            }
            
            return ocean_traits
            
        except Exception as e:
            st.error(f"Error during prediction: {e} \nPlease choose another image or refresh your browser")
            return None

def main():
    st.title("🖋️ Handwriting OCEAN Psychology Traits Mapper")
    st.markdown("Upload your handwriting to analyze **Big Five** personality traits **(DUMMY VIEW. NEED AI MODEL)**")
    
    # Load model (dummy)
    model_data = load_model()
    if model_data is None:
        st.stop()
    
    model, processor = model_data
    
    # Personality traits
    personality_traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    
    # Input method selection
    st.subheader("Choose Input Method")
    input_method = st.radio(
        "How would you like to provide your handwriting sample?",
        ["Upload Image File", "Webcam input V1"],
        horizontal=True
    )
    
    uploaded_image = None
    
    if input_method == "Upload Image File":
        # File uploader
        uploaded_file = st.file_uploader(
            "insert your handwriting or click to browse",
            type=['png', 'jpg', 'jpeg'],
            help="Supported: PNG, JPG, JPEG | Max size: 10MB | Use **clear images** for best results"
        )
        
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            
    elif input_method == "Webcam input V1":
        # Camera input
        camera_photo = st.camera_input(
            "Take a photo of your handwriting",
            help="Make sure your handwriting is well-lit and clearly visible in the frame"
        )
        
        if camera_photo is not None:
            uploaded_image = Image.open(camera_photo)
    
    # Process the image (regardless of source)
    if uploaded_image is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_image, caption="**PLEASE MAKE SURE THE IMAGE IS WELL-LIT AND READABLE**", use_container_width=True)
            
            # Image info
            file_size_mb = 0  # Camera input doesn't have direct file size
            if input_method == "Upload Image File" and uploaded_file is not None:
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.info(f"**Image size:** {uploaded_image.size[0]} x {uploaded_image.size[1]} pixels | **File size:** {file_size_mb:.2f}MB")
            else:
                st.info(f"**Image size:** {uploaded_image.size[0]} x {uploaded_image.size[1]} pixels | **Source:** Camera")
        
        with col2:
            # Update button text based on model type
            model, processor = model_data
            button_text = "RUN VIT MODEL" if model != "dummy_model" else "RUN DUMMY OUTPUT"
            model_status = "ViT initialized" if model != "dummy_model" else "⚠️ Dummy Mode"
            
            st.info(model_status)
            
            if st.button(button_text, type="primary", use_container_width=True):
                with st.spinner("Analyzing your handwriting... Please wait"):
                    try:
                        # Preprocess image
                        processed_inputs = preprocess_image(uploaded_image, processor)
                        
                        if processed_inputs is None:
                            st.error("Failed to preprocess image")
                            st.stop()
                        
                        # Make prediction
                        personality_results = predict_personality(processed_inputs, model)
                        
                        if personality_results is None:
                            st.error("Failed to make prediction")
                            st.stop()
                        
                        # Display results
                        success_message = "✨ Analysis Complete!" if model != "dummy_model" else "DUMMY Complete!"
                        st.success(success_message)
                        st.subheader("Personality Analysis Results")
                        
                        # Create metrics display
                        for trait in personality_traits:
                            prob = personality_results[trait]
                            col_metric, col_bar = st.columns([1, 2])
                            
                            with col_metric:
                                st.metric(
                                    label=trait,
                                    value=f"{prob:.1%}",
                                    delta=None
                                )
                            
                            with col_bar:
                                st.progress(prob)
                        
                        # Dominant trait
                        dominant_trait = max(personality_results, key=personality_results.get)
                        dominant_prob = personality_results[dominant_trait]
                        
                        st.success(f"🌟 **Dominant trait:** {dominant_trait} ({dominant_prob:.1%})")
                        
                        # Show raw results for debugging
                        with st.expander("📊 Detailed Results"):
                            st.json(personality_results)
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                        st.error("Please check your image and try again.")
                        # Optionally show more detailed error info in debug mode
                        if st.checkbox("Show detailed error info"):
                            st.exception(e)
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This program uses TensorFlow based **Vision Transformer (ViT)** model to analyze handwriting 
        and map personality traits based on Warren Norman's **Big Five** model:
        
        - **Openness** to new experiences
        - **Conscientiousness** on discipline and organization  
        - **Extraversion** towards social situations
        - **Agreeableness** in relationships
        - **Neuroticism** towards emotional stability
        """)
        
        st.header("📸 Camera Tips")
        st.markdown("""
        **For best camera results:**
        - Hold your device steady
        - Ensure good lighting (avoid shadows)
        - Position handwriting flat and straight
        - Fill the frame with your writing
        - Avoid glare or reflections
        """)
        
        st.header("⚠️**PLEASE USE WELL-LIT IMAGES**")
        st.markdown("""
        - Ensure good lighting and contrast
        - Include multiple words or sentences
        - Avoid blurry or rotated images if possible
        """)
        
        st.header("🔧 Changelog")
        st.success("**V0.10** ViT AI Model integration")
        st.warning("""
        **V0.01** Added camera input support
        **V0.00** Dummy mode with simulated results. AI model yet to be integrated
        """)

if __name__ == "__main__":
    main()