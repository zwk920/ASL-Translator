import tensorflow as tf
from keras import layers, Model
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model

print("Built with GPU support:", tf.test.is_built_with_gpu_support())
print("GPU devices available:", tf.config.list_physical_devices('GPU'))

def letter_to_number(string):
    """
    Converts each letter in a string to its corresponding number position in the alphabet.
    Returns a new string with the converted letters.
    """
    # Create a dictionary to map each letter to its corresponding number position
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    letter_to_number_dict = {letter: index+1 for index, letter in enumerate(alphabet)}
    
    # Convert each letter in the string to its corresponding number position
    new_list = []
    for char in string:
        if char.lower() in letter_to_number_dict:
            new_list.append(letter_to_number_dict[char.lower()])
        else:
            new_list.append(char)
    
    return new_list[0]-1

def letters_to_numbers_list(string_list):
    """
    Converts each letter in each string in a list to its corresponding number position in the alphabet.
    Returns a new list of strings with the converted letters.
    """
    new_list = []
    for string in string_list:
        new_list.append(letter_to_number(string))
    
    return new_list
train_labels = np.load("data/train_labels.npy")
X_train = np.load("data/train_coords.npy")
val_labels = np.load("data/val_labels.npy")
X_val = np.load("data/val_coords.npy")
test_labels = np.load("data/test_labels.npy")
X_test = np.load("data/test_coords.npy")


y_train = to_categorical(letters_to_numbers_list(train_labels), num_classes=26)
y_test = to_categorical(letters_to_numbers_list(test_labels), num_classes=26)
y_val = to_categorical(letters_to_numbers_list(val_labels), num_classes=26)

filename = "classes.txt"
with open(filename) as file:
    class_names = [line.rstrip() for line in file]

# Define the input shape
input_shape = (21, 3)

# Define the model architecture
inputs = layers.Input(shape=(21, 3))
x = layers.Conv1D(64, 1, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(128, 1, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(1024, 1, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(26, activation='softmax')(x)

#Define the optimizer
opt = tf.keras.optimizers.Adam(lr=0.0001)

model = Model(inputs, outputs)
Model.summary(model)
#Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Fit the model with data augmentation
history = model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val))

#Plot the training and validation accuracy
f = plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
f.savefig("accuracy.pdf", bbox_inches='tight')

f = plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')
f.savefig("loss.pdf", bbox_inches='tight')

#Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
model.save("ASL")

print('Test accuracy:', test_acc)