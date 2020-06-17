import numpy as np
from keras.models import model_from_json
from model import *
from constants import *
from dataIO import *
from visualize import *

t, v = splitData(n_images=3)

X, Y = next(t)
print(np.array(X).shape)

input = X[0].reshape(output_shape)
groundTruth = Y[0].reshape(output_shape)
plot(np.nonzero(input), 'input.html')
plot(np.nonzero(groundTruth), 'groundTruth.html')


with open('model.json', 'r') as f:
        model_file = f.read()
model = model_from_json(model_file)
model.load_weights('model.h5')
model.compile(loss='mse', optimizer='adam')

gen = scale(model.predict(X).squeeze())
#gen = model.predict(X).squeeze()
np.save('generated.npy', gen)

data1 = gen > 0.9
plot(np.nonzero(data1), 'generated1.html')


data2 = gen > 0.5
plot(np.nonzero(data2), 'generated2.html')
