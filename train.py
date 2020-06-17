import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger

from model import *
from constants import *
from dataIO import *

#checkpoint_path = "checkpoints/checkpoint.ckpt"
#cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                 save_weights_only=True,
#                                                 verbose=1)

training, validation = splitData(0.9)
model = build_model_2()

#logger = CSVLogger('log.csv', append=True, separator=',')
model.fit_generator(generator=training, epochs=n_epochs, steps_per_epoch=n_steps_per_epoch)

model_json = model.to_json()
with open("model.json", "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
