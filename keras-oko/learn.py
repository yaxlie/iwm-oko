from __future__ import print_function
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from imageio import imread
from sklearn.utils import shuffle
import matplotlib.pylab as plt
import numpy as np
import glob

batch_size = 128
num_classes = 2
epochs = 500

# input image dimensions
img_x, img_y = 64, 64

input_shape = (64, 64, 3)

def load_data(path):
	true_lst=glob.glob(path+"/true/*.jpg")
	false_lst=glob.glob(path+"/false/*.jpg")
	lst=true_lst+false_lst
	
	set_im=np.array([ imread(img) for img in lst])
	set_lb= np.zeros(len(lst))
	set_lb[0:len(true_lst)]=1.0
	return shuffle(set_im,set_lb,random_state=2)

(x_train,y_train) = load_data("train")
(x_test,y_test) = load_data("test")
# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(16, kernel_size=(8, 8), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (8, 8), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# zapisz model do pliku yaml
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# zapisz wagi do pliku h5
model.save_weights("model.h5")
print("Saved model to disk")

plt.plot(range(1, epochs+1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

#tak się korzysta z wyuczonej sieci
#print(model.predict(x_test,100))
