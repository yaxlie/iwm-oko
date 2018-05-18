from __future__ import print_function
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import model_from_yaml
import numpy as np
import glob

# Załaduj model
yaml_file = open('model.yaml', 'r')
model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(model_yaml)
# Załaduj wagi
model.load_weights("model.h5")

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# tego można użyć do zbadania wielu obrazków na raz
# zmienna test powinna mieć strukturę 4 wymiarowej tablicy numpy
# (sample_number, x_img_size, y_img_size, num_channels)
def CheckImages(test):
	result=[]
	for x in model.predict(test):
		if x[0]>x[1]:
			result.append(False)
		else:
			result.append(True)
	return result
