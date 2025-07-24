import tensorflow as tf

DIR = r"C:\collage\feasiable\92%_acc.keras"


model = tf.keras.models.load_model(DIR, compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("MobilenetV2.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Converted and quantized model saved as 'efficientnet96_quant.tflite'")