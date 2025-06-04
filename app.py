import streamlit as st
import tensorflow as tf
from transformers import TFViTForImageClassification, ViTImageProcessor
import numpy as np
from PIL import Image
import os
import glob

# Page configuration
st.set_page_config(
    page_title="Handwriting Personality Analysis",
    page_icon="🖋️",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained ViT model"""
    # Find the latest saved model
    model_dirs = glob.glob("trained_vit_model_perfect/personality_vit_*")
    if not model_dirs:
        st.error("No trained model found! Please run the training notebook first.")
        return None
    
    latest_model_dir = max(model_dirs)
    
    try:
        model = TFViTForImageClassification.from_pretrained(latest_model_dir)
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, processor):
    """Preprocess image for ViT model"""
    # Resize to expected size
    img_height = processor.size["height"]
    img_width = processor.size["width"]
    
    image = image.resize((img_width, img_height))
    image_array = np.array(image) / 255.0
    
    # Handle grayscale images
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[-1] == 4:  # RGBA
        image_array = image_array[:, :, :3]  # Remove alpha channel
    
    # Transpose to channels first (as done in training)
    image_array = np.transpose(image_array, (2, 0, 1))
    image_batch = image_array[np.newaxis, ...]
    
    return image_batch

def main():
    st.title("🖋️ Handwriting Personality Analysis")
    st.markdown("Upload a handwriting sample to analyze **Big Five** personality traits using Vision Transformer AI")
    
    # Load model
    model_data = load_model()
    if model_data is None:
        st.stop()
    
    model, processor = model_data
    
    # Personality traits
    personality_traits = ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"]
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Drop your handwriting image here or click to browse",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of handwriting for best results"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="📝 Your handwriting sample", use_column_width=True)
            
            # Image info
            st.info(f"**Image size:** {image.size[0]} x {image.size[1]} pixels")
        
        with col2:
            if st.button("🔍 Analyze Personality", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your handwriting patterns..."):
                    try:
                        # Preprocess image
                        processed_image = preprocess_image(image, processor)
                        
                        # Make prediction
                        predictions = model.predict(processed_image, verbose=0)
                        probabilities = tf.nn.softmax(predictions.logits).numpy()[0]
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        st.subheader("🎯 Personality Analysis Results")
                        
                        # Create metrics display
                        for i, (trait, prob) in enumerate(zip(personality_traits, probabilities)):
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
                        dominant_trait_idx = np.argmax(probabilities)
                        dominant_trait = personality_traits[dominant_trait_idx]
                        dominant_prob = probabilities[dominant_trait_idx]
                        
                        st.success(f"🌟 **Dominant trait:** {dominant_trait} ({dominant_prob:.1%})")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                        st.error("Please check your image and try again.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses a **Vision Transformer (ViT)** model to analyze handwriting 
        and predict personality traits based on the **Big Five** model:
        
        - **Agreeableness**: Compassionate and cooperative
        - **Conscientiousness**: Organized and disciplined  
        - **Extraversion**: Outgoing and energetic
        - **Neuroticism**: Anxious and emotionally reactive
        - **Openness**: Creative and open to new experiences
        """)
        
        st.header("📋 Tips for Best Results")
        st.markdown("""
        - Use clear, high-quality images
        - Ensure good lighting and contrast
        - Include multiple words or sentences
        - Avoid blurry or rotated images
        """)

if __name__ == "__main__":
    main()