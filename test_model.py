import tensorflow as tf

print("TensorFlow imported", flush=True)

model = tf.keras.models.load_model("brain_tumor_model.h5")

print("Model loaded successfully", flush=True)