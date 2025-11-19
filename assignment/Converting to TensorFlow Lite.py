import os
import tensorflow as tf

OUT_DIR = os.path.dirname(__file__) or "."
candidates = ["recycle_model.h5", "best_model.h5", os.path.join("final_model")]
model_path = next((os.path.join(OUT_DIR, p) for p in candidates if os.path.exists(os.path.join(OUT_DIR, p))), None)
if model_path is None:
    raise FileNotFoundError("No Keras model found. Expected one of: recycle_model.h5, best_model.h5, final_model/")

# load model (handles both .h5 and SavedModel dir)
if os.path.isdir(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

try:
    tflite_model = converter.convert()
except Exception as e:
    # fallback: try conversion without optimizations to get a working .tflite and surface error
    print("Optimized conversion failed:", e)
    print("Retrying conversion without optimizations...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

out_path = os.path.join(OUT_DIR, "recycle_model.tflite")
with open(out_path, "wb") as f:
    f.write(tflite_model)

print("Conversion completed:", out_path)
