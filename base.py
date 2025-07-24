import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

IMG_SIZE = 96
BATCH_SIZE = 24
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
UNFREEZE_LAYERS = -20
GAMMA = 1.0
ALPHA = 0.25
DATA_DIR = "AIDER_full"

log_dir = "logs/fit/" + tf.timestamp().numpy().astype(str)
summary_writer = tf.summary.create_file_writer(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
import io

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    return tf.expand_dims(image, 0)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.1,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.1,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.1)
])

def augment(image, label):
    return data_augmentation(image), label

train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    alpha=0.35
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

from tensorflow.keras.losses import Loss

class FocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.math.pow(1 - y_pred, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=-1))

focal = FocalLoss(gamma=GAMMA, alpha=ALPHA)

def make_scheduler():
    lr_rate = 0.625

    def scheduler(epoch, lr):
        nonlocal lr_rate
        if epoch > 0 and epoch % 2 == 0:
            lr_rate += 0.1
            return lr * lr_rate
        return lr
    return scheduler


lr_callback = tf.keras.callbacks.LearningRateScheduler(make_scheduler())


y_train = np.concatenate([y.numpy() for x, y in train_ds])
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}
print("Class Weights:", class_weights_dict)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=focal,
    metrics=["accuracy"]
)

print("\n🔁 Phase 1: Training classifier head...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        lr_callback,
        tensorboard_callback
    ]
)

for layer in base_model.layers[:UNFREEZE_LAYERS]:
    layer.trainable = False
for layer in base_model.layers[UNFREEZE_LAYERS:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=focal,
    metrics=["accuracy"]
)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-7
)

print("\n🔧 Phase 2: Fine-tuning last layers...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        lr_callback,
        reduce_lr_callback,
        tensorboard_callback
    ]
)

y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

model.save("mobilenetv2_disaster_classification_finetuned.keras")
print("\n✅ Model saved as: mobilenetv2_disaster_classification_finetuned.keras")

plt.figure(figsize=(8, 6))
sns.heatmap(cm_percentage, annot=True, fmt='.2f',
            xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix (%)')
plt.tight_layout()
plt.show()

figure = plt.figure(figsize=(8, 6))
sns.heatmap(cm_percentage, annot=True, fmt='.2f',
            xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix (%)')
plt.tight_layout()

cm_image = plot_to_image(figure)
cm_image = plot_to_image(figure)

with summary_writer.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=0)