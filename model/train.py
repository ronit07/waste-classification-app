"""
Fine-tunes MobileNetV2 on a waste classification dataset.
Dataset structure expected:
  data/
    train/
      recyclable/
      organic/
      landfill/
    val/
      recyclable/
      organic/
      landfill/

Download a suitable dataset from Kaggle (e.g. "Waste Classification data" by Sashaank Sekar).
Run: python model/train.py
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

IMG_SIZE  = (224, 224)
BATCH     = 32
EPOCHS    = 20
CLASSES   = ['recyclable', 'organic', 'landfill']
NUM_CLASSES = len(CLASSES)

def build_model():
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False  # freeze base layers initially

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(NUM_CLASSES, activation='softmax')(x)

    return Model(base.input, out)

def get_generators():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=[0.8, 1.2],
        zoom_range=0.15,
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_ds = train_gen.flow_from_directory(
        'data/train', target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='categorical', classes=CLASSES,
    )
    val_ds = val_gen.flow_from_directory(
        'data/val', target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='categorical', classes=CLASSES,
    )
    return train_ds, val_ds

def train():
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    train_ds, val_ds = get_generators()

    os.makedirs('model', exist_ok=True)
    callbacks = [
        EarlyStopping(patience=4, restore_best_weights=True, monitor='val_accuracy'),
        ModelCheckpoint('model/waste_model.h5', save_best_only=True, monitor='val_accuracy'),
    ]

    print("Phase 1: Training head with frozen base...")
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    # Phase 2: Unfreeze top layers for fine-tuning
    print("Phase 2: Fine-tuning top layers...")
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              initial_epoch=10, callbacks=callbacks)

    print(f"Training complete. Best model saved to model/waste_model.h5")

if __name__ == '__main__':
    train()
