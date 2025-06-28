import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Get working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/trained_fashion_mnist_model.h5"

# Load model without compiling (to avoid legacy issues)
model = tf.keras.models.load_model(model_path, compile=False)

# Manually compile
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Class labels
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Detailed descriptions
item_details = {
    'T-shirt/top': {
        'emoji': '👕',
        'desc': 'A lightweight, casual upper-body wear.',
        'price': '₹300 - ₹1000',
        'material': 'Cotton, Polyester',
        'usage': 'Casual, Daily Wear'
    },
    'Trouser': {
        'emoji': '👖',
        'desc': 'Bottom wear for formal and casual settings.',
        'price': '₹700 - ₹2500',
        'material': 'Cotton, Denim, Wool',
        'usage': 'Formal, Office, Everyday'
    },
    'Pullover': {
        'emoji': '🧶',
        'desc': 'Warm sweater pulled over the head.',
        'price': '₹1000 - ₹3000',
        'material': 'Wool, Fleece, Acrylic',
        'usage': 'Winter, Casual, Cozy Wear'
    },
    'Dress': {
        'emoji': '👗',
        'desc': 'One-piece stylish garment for women/girls.',
        'price': '₹800 - ₹5000',
        'material': 'Cotton, Silk, Georgette',
        'usage': 'Party, Casual, Formal'
    },
    'Coat': {
        'emoji': '🧥',
        'desc': 'Outerwear for warmth and style.',
        'price': '₹1500 - ₹7000',
        'material': 'Wool, Leather, Tweed',
        'usage': 'Winter, Formal, Outdoor'
    },
    'Sandal': {
        'emoji': '🩴',
        'desc': 'Open-toe, breathable summer footwear.',
        'price': '₹200 - ₹1500',
        'material': 'Rubber, Leather, Foam',
        'usage': 'Casual, Beach, Everyday'
    },
    'Shirt': {
        'emoji': '👔',
        'desc': 'Collared button-down upper-body wear.',
        'price': '₹600 - ₹2500',
        'material': 'Cotton, Linen, Polyester',
        'usage': 'Formal, Semi-Formal, Office'
    },
    'Sneaker': {
        'emoji': '👟',
        'desc': 'Sporty, comfortable and stylish shoes.',
        'price': '₹1000 - ₹6000',
        'material': 'Canvas, Mesh, Rubber',
        'usage': 'Casual, Sports, Travel'
    },
    'Bag': {
        'emoji': '👜',
        'desc': 'Accessory for carrying personal items.',
        'price': '₹400 - ₹3000',
        'material': 'Leather, Canvas, Nylon',
        'usage': 'Shopping, Travel, Daily Use'
    },
    'Ankle boot': {
        'emoji': '🥾',
        'desc': 'Boots that cover the ankle; stylish and warm.',
        'price': '₹1200 - ₹6000',
        'material': 'Leather, Suede, Synthetic',
        'usage': 'Winter, Fashion, Outdoor'
    }
}

# Image preprocessing
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L')  # grayscale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

# Streamlit UI
st.set_page_config(page_title="Fashion Item Classifier 👗🧥", layout="centered")
st.title('🧠 Fashion Item Classifier 👕👖👟')

uploaded_image = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img, caption="🖼️ Uploaded Image", use_column_width=True)

    with col2:
        if st.button('🔍 Classify'):
            img_array = preprocess_image(uploaded_image)
            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]
            confidence = tf.nn.softmax(result)[0][predicted_class].numpy()

            # Get item details
            details = item_details[prediction]

            st.success(f"{details['emoji']} **Prediction**: {prediction}")
            st.markdown(f"**📝 Description**: {details['desc']}")
            st.markdown(f"**💰 Price Range**: {details['price']}")
            st.markdown(f"**🧵 Material**: {details['material']}")
            st.markdown(f"**👥 Usage**: {details['usage']}")
            st.info(f"📊 **Confidence**: {confidence * 100:.2f}%")
