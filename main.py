
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps, ImageFilter


# Cache the model so Streamlit doesn’t retrain every time you reload the app
# This is because in Streamlit, every time you reload or interact with the app (like pressing a button, changing a slider, etc.), the script runs from top to bottom again.
@st.cache_resource
def load_model():
    # ---------------------------
    # 1. Load the MNIST dataset
    # ---------------------------
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # x_train: training images (60,000 samples, shape 28×28)
    # y_train: training labels (digits 0–9)
    # x_test, y_test: test set (10,000 samples)

    # ---------------------------
    # 2. Normalize pixel values
    # ---------------------------
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Converts values from [0, 255] → [0, 1] (easier for neural networks to process)
    # In the MNIST dataset (and most image datasets), each pixel value is an integer between 0 and 255:
        # 0 → black
        # 255 → white
        # values in between → shades of gray

    # ---------------------------
    # 3. Add channel dimension
    # ---------------------------
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # MNIST images are grayscale (28×28). CNN expects 3D input: (height, width, channels).
    # Each pixel can have one or more color values, and each such "layer" is called a channel.
    # Channel = how many numbers describe the color of a single pixel. 1 means any number from 0-255. 
    # In RGB the channel will be 3 one for each Red Green and Blue. ([0-255] [0-255] [0-255])
    # expand_dims adds a new axis at the end → shape becomes (28, 28, 1).


    # ---------------------------
    # 4. Build CNN Model
    # ---------------------------
    model = keras.Sequential([      # Sequential = stack layers one after another
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        # Conv2D: Convolutional layer for 2D images. It applies 32 filters of size 3×3 to detect patterns (edges, curves, etc.)
        # Filters (or kernels) = small sliding windows that look for patterns in the image.
        # 32 filters = the layer will learn 32 different kinds of patterns.
        # Size 3×3 = each filter looks at a small patch of 3×3 pixels at a time.
        # activation="relu": introduces non-linearity. Without activation, a neural network would just be linear (like a straight line equation).
        # input_shape=(28,28,1): image shape (H, W, channels)

        layers.MaxPooling2D((2, 2)),    # Pooling downsamples the image (reduces size) to improve the performance by Keeping important info
        
        layers.Flatten(),           # Flatten: converts 2D feature maps → 1D vector (for Dense layers)
        # Convolutions & pooling → extract features (edges, shapes, textures).
        # Flatten → turns those features into a vector.
        # Dense layers → combine those features to make the final decision (e.g., "This is a cat").
        
        layers.Dense(128, activation="relu"),
        # Dense: fully connected layer with 128 neurons
        # activation="relu": helps capture complex patterns

        layers.Dense(10, activation="softmax"),
        # Dense: output layer with 10 neurons (one for each digit 0–9)
    ])

    # ---------------------------
    # 5. Compile Model
    # ---------------------------
    model.compile(
        optimizer="adam",                       # Adam = adaptive gradient optimizer                   
        loss="sparse_categorical_crossentropy", # loss for multi-class classification
        metrics=["accuracy"]                    # track accuracy during training
    )

    # ---------------------------
    # 6. Train the Model
    # ---------------------------
    model.fit(
        x_train, y_train,                       # training data
        validation_data=(x_test, y_test),       # This is data the model never sees during training, used only for evaluation after each epoch.
        epochs=1,                               # train for 1 full pass through dataset
        # Epoch = one full pass through the training dataset.
        # If epochs=1, the model sees all 60,000 MNIST images once.
        # If epochs=10, the model will go through the dataset 10 times.
        batch_size=128,                         # number of samples per gradient update
        # Instead of feeding all 60,000 images at once, we split into mini-batches.
        # Each batch → 128 samples.
        # The model updates weights once per batch (called one step).
        verbose=1                               # Controls training output display. 1 = show progress bar. For example: 469/469 [==============================] - 10s 20ms/step - loss: 0.2554 - accuracy: 0.9271 - val_loss: 0.0897 - val_accuracy: 0.9730
    )
    
    # ---------------------------
    # 7. Return trained model
    # ---------------------------
    return model

model = load_model()


# ---------------------------
# Streamlit App Title and Canvas
# ---------------------------
st.title("MNIST Digit Classifier")
st.write("Draw a digit (0–9) in the canvas below and let the model predict it!")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)


# ---------------------------
# Preprocessing to better match MNIST
# ---------------------------
def preprocess_canvas_image(image_data):
    img = Image.fromarray(image_data.astype(np.uint8)).convert("L")
    img = ImageOps.autocontrast(img)

    arr = np.array(img)
    arr[arr < 10] = 0

    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = arr[y0:y1+1, x0:x1+1]

    h, w = crop.shape
    s = max(h, w) + 8
    square = np.zeros((s, s), dtype=np.uint8)
    y_off = (s - h) // 2
    x_off = (s - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = crop

    square = Image.fromarray(square).filter(ImageFilter.GaussianBlur(radius=1))
    square = square.resize((28, 28), Image.LANCZOS)

    arr28 = np.array(square).astype("float32") / 255.0
    if arr28.mean() > 0.5:
        arr28 = 1.0 - arr28

    return np.expand_dims(arr28, (0, -1))


# ---------------------------
# Button to Trigger Prediction
# ---------------------------
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img_array = preprocess_canvas_image(canvas_result.image_data)
        if img_array is None:
            st.warning("Please draw a digit before predicting.")
        else:
            st.image((img_array[0, :, :, 0] * 255).astype("uint8"), caption="Preprocessed 28×28", width=96, clamp=True)
            prediction = model.predict(img_array)
            predicted_class = int(np.argmax(prediction))
            st.subheader(f"Predicted Digit: **{predicted_class}**")
            st.bar_chart(prediction[0])
    else:
        st.warning("Please draw a digit before predicting.")
