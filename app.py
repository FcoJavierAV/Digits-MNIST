import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar modelo en Streamlit
model = load_model('mnist_model.keras')

# Crear la interfaz de usuario
st.title("Clasificación de Dígitos MNIST")
st.write("Sube una imágen para identificar el número.")

uploaded_file = st.file_uploader("Sube una imagen en escala de grises 28x28", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Procesar la imágen
    image = Image.open(uploaded_file).convert('L') # Convertir RGB a ByN
    image = image.resize((28, 28))
    image = np.array(image).astype('float32') / 255 - 0.5# Normalizar
    mage = image.reshape(1, 784)

    # Motrar la imágen subida
    st.image(image, caption='Imágen subida', use_column_width=True)

    # Predicción
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    
    st.image(uploaded_file, caption=f"Predicción: {predicted_class}", use_column_width=True)
