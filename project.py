import pandas as pd
import numpy as np
import preprocessing # type: ignore
import time as time
import tensorflow as tf
#from keras.layers import experimental
import matplotlib.pyplot as plt

start_time = time.time()

def build_cnn_model(input_shape, num_classes):
    model = tf.keras.Sequential()

    #add normalizing layer
    #model.add(tf.keras.layers.experimental.preprocessing.Normalization())

    #add convolutional and pooling layers
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=1))

    #flatten and fully connected layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    # Add the output layer with softmax activation for multi-class classification
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


train_files = ['101','106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230'] #training files numbers
test_files = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'] #testing files numbers

annotation_classes = {
    'N': 0, 'L': 0, 'R': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1, 'e': 1, 'j': 1,
    'V': 2, 'E': 2,
    'F': 3,
    'P': 4, '/': 4, 'f': 4, 'u': 4,
    '+' : 0,
    '[' : 0,
    '!' : 0,
    ']' : 0,
    'x' : 0,
    'Q' : 0,
    '~' : 0,
    '|' : 0
}

x_train = []
y_train = []

x_test = []
y_test = []

#get data from files as dataframe and make big 3d array for training
################## make training arrays ##################
for record_name in train_files:

  #set up file paths
  file_path_x = f"C:\\Users\\rigga\\Documents\\BMEN 207\\Honors project\\{record_name}_features.csv"

  #read in data from files as dataframe
  data = pd.read_csv(file_path_x)

  x_data = data.copy()
  x_data.pop('index1')
  x_data.pop('index2')
  x_data.pop('notes')  

  y_data = data['notes'].copy().map(annotation_classes)

  x_arr = x_data.to_numpy().tolist()
  y_arr = y_data.to_numpy().tolist()

  x_train = x_train + x_arr
  y_train = y_train + y_arr
  
x_train = np.array(x_train)
y_train = np.array(y_train)
##########################################################

################## make testing arrays ##################
for record_name in test_files:

  #set up file paths
  file_path_x = f"C:\\Users\\rigga\\Documents\\BMEN 207\\Honors project\\{record_name}_features.csv"

  #read in data from files as dataframe
  data = pd.read_csv(file_path_x)

  x_data = data.copy()
  x_data.pop('index1')
  x_data.pop('index2')
  x_data.pop('notes')  

  y_data = data['notes'].copy().map(annotation_classes)

  x_arr = x_data.to_numpy().tolist()
  y_arr = y_data.to_numpy().tolist()

  x_test = x_test + x_arr
  y_test = y_test + y_arr
  
x_test = np.array(x_test)
y_test = np.array(y_test)

##########################################################
# Reshape input data to match the expected shape of the model

# Verify the shapes of the input data before training

print("Shape of x_train:", x_train.shape)
print("Shape of x_test:", x_test.shape)

model = build_cnn_model((1,9),5)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  
    patience=3,  
    restore_best_weights=True  
)

x_train = np.reshape(x_train, (len(x_train),1,9))
x_test = np.reshape(x_test, (len(x_test),1,9))

print(x_train.shape)
history = model.fit(x_train, y_train, epochs=10, validation_data = (x_test,y_test), callbacks=[early_stopping_callback])

plot_training_history(history)

end_time = time.time()
print(f'model took {end_time - start_time} to train')
