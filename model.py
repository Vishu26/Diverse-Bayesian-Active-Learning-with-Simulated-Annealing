from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

def LeNet(input_shape):
    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                        activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu', name='embedding'))
    model.add(Dropout(0.55))
    model.add(Dense(10, activation='softmax'))

    opt = Adam(learning_rate=0.001)
    model.compile(loss=categorical_crossentropy, 
                optimizer=opt, 
                metrics=['accuracy']) 

    return model