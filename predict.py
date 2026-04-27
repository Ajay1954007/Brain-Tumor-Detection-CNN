import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

print("Loading model...", flush=True)
model = tf.keras.models.load_model(
    "brain_tumor_model.h5",
    compile=False
)
print("Model loaded.", flush=True)

classes = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]

img_path = "test.jpg"
print("Loading image...", flush=True)

img = image.load_img(img_path, target_size=(224,224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

print("Predicting...", flush=True)
prediction = model.predict(img)

result = classes[np.argmax(prediction)]

print("Prediction:", result, flush=True)