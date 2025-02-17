import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from preprocess import train_generator, valid_generator  # Import data loaders

# Define CNN Model
model = Sequential([
    # Convolutional Block 1
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    # Convolutional Block 2
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    # Convolutional Block 3
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    # Flattening Layer
    Flatten(),

    # Fully Connected Layers
    Dense(128, activation="relu"),
    Dropout(0.5),  # Prevent Overfitting
    Dense(64, activation="relu"),

    # Output Layer (3 classes)
    Dense(3, activation="softmax")
])

# Compile the Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Display Model Summary
model.summary()

# Train the model
EPOCHS = 15
history = model.fit(train_generator, validation_data=valid_generator, epochs=EPOCHS)

# Save the trained model
model.save("skin_type_model.h5")