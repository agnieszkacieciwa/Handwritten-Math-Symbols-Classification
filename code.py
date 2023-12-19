import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import numpy as np
import matplotlib.pyplot as plt

# Define data directories
dataset_dir = 'C:/Users/agnie/OneDrive/Pulpit/dataset'

# Set parameters for data augmentation and normalization
batch_size = 32
img_height, img_width = 100, 100

# Create ImageDataGenerator with validation split
datagen = ImageDataGenerator(
    rescale=1./255,        # Normalize pixel values to [0,1]
    shear_range=0.2,       
    zoom_range=0.2,        
    horizontal_flip=True,  
    validation_split=0.2,   # 20% of the data will be used for validation
)

# Use flow_from_directory to load data from the directory
train_iterator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  
    subset='training',         
)

test_iterator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',       
)

# Set up model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten layer to transition from convolutional to dense layers
model.add(Flatten())

# Dense layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Add dropout for regularization

# Output layer
model.add(Dense(19, activation='softmax'))

# Create a learning rate schedule
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
)

# Compile the model with the learning rate schedule
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Define callbacks
checkpoint_callback = ModelCheckpoint('model_checkpoint.h5', save_best_only=True)
early_stopping_callback = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr_callback = ReduceLROnPlateau(factor=0.1, patience=4, min_lr=1e-6)
tensorboard_callback = TensorBoard(log_dir='logs')

# Train the model
history = model.fit(
    train_iterator,
    steps_per_epoch = train_iterator.samples // train_iterator.batch_size,
    validation_data = test_iterator,
    validation_steps = test_iterator.samples // test_iterator.batch_size,
    epochs = 10,
    callbacks = [checkpoint_callback, early_stopping_callback, reduce_lr_callback, tensorboard_callback],
    verbose=1
)

# Evaluate the model on the test set
evaluation_results = model.evaluate(test_iterator, steps = test_iterator.samples // test_iterator.batch_size)
print("Test Loss:", evaluation_results[0])
print("Test Accuracy:", evaluation_results[1])
if len(evaluation_results) > 2:
    print("Test AUC:", evaluation_results[2])

# Visualize the history
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()

# Save the model
model.save('handwritten_math_symbols_model.h5')
