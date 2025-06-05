from transformers import ViTImageProcessor, TFViTForImageClassification
from PIL import Image
import numpy as np

MODEL_NAME = "google/vit-base-patch16-224"

# Load the model and processor
# processor = ViTImageProcessor.from_pretrained("personality_vit_20250521-144334")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
loaded_model = TFViTForImageClassification.from_pretrained("personality_vit_20250521-144334")

# Load the image
image = Image.open("test.jpg")

# # Preprocess the image
inputs = processor(images=image, return_tensors="tf")

print("masuk")
# Make prediction
predictions = loaded_model(**inputs)
predicted_class_idx = np.argmax(predictions.logits, axis=-1)[0]
predicted_class = loaded_model.config.id2label[predicted_class_idx]

# Convert logits to probabilities
probabilities = np.exp(predictions.logits) / np.sum(np.exp(predictions.logits))

# OCEAN traits based on actual model's id2label mapping
# Order from model config.json: 0:Agreeableness, 1:Conscientiousness, 2:Extraversion, 3:Neuroticism, 4:Openness
ocean_traits = {
    'Agreeableness': float(probabilities[0][0]),
    'Conscientiousness': float(probabilities[0][1]),
    'Extraversion': float(probabilities[0][2]),
    'Neuroticism': float(probabilities[0][3]),
    'Openness': float(probabilities[0][4])
}

print(ocean_traits)
print("\nOCEAN Personality Traits:")
for trait, score in ocean_traits.items():
    print(f"{trait}: {score:.4f}")
print(f"Predicted class: {predicted_class}")