import streamlit as st
import numpy as np
from PIL import Image
import random
import time

# Page configuration
st.set_page_config(
    page_title="Handwriting Personality Analysis",
    page_icon="🖋️",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained ViT model (DUMMY VERSION)"""
    # Simulate model loading delay
    time.sleep(1)
    return "dummy_model", "dummy_processor"

def preprocess_image(image, processor):
    """Preprocess image for ViT model (DUMMY VERSION)"""
    # Just return the image as-is for display purposes
    return np.array(image)

def predict_personality(image_array, model):
    """
    Dummy prediction function that returns random personality percentages
    Returns: dictionary with personality traits and their percentages
    """
    # Simulate processing time
    time.sleep(2)
    
    # Generate random percentages that sum to 100%
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    
    # Generate random values and normalize to percentages
    raw_values = [random.uniform(0.1, 0.4) for _ in traits]
    total = sum(raw_values)
    percentages = [(val / total) for val in raw_values]
    
    # Create result dictionary
    results = {}
    for trait, percentage in zip(traits, percentages):
        results[trait] = percentage
    
    return results

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
            help="Supported: PNG, JPG, JPEG | 📏 Max size: 10MB | Use **clear images** for best results"
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
            if st.button("RUN DUMMY OUTPUT", type="primary", use_container_width=True):
                with st.spinner("Processing... Please wait"):
                    try:
                        # Preprocess image (dummy)
                        processed_image = preprocess_image(uploaded_image, processor)
                        
                        # Make prediction (dummy)
                        personality_results = predict_personality(processed_image, model)
                        
                        # Display results
                        st.success("DUMMY Complete!")
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
        st.warning("""
        **V0.1** Added camera input support
        **V0.0** Dummy mode with simulated results. AI model yet to be integrated
        """)

if __name__ == "__main__":
    main()