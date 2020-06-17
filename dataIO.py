import numpy as np
import os
from PIL import Image
from visualize import *
from scipy.ndimage import rotate, zoom
from skimage.transform import resize
import matplotlib.pyplot as plt
import multiprocessing.dummy as mp
from constants import *

def visualHull(images, angles):
        hull = np.full(output_shape, 1)
        for i in range(len(images)):
                angle = angles[i] if i == 0 else angles[i] - angles[i-1]
                image = resize(images[i], input_shape[:2]) == 0
                hull = rotate(hull, angle, mode='nearest', axes=(1,2), reshape=False)
                mask = np.dstack((image,) * output_shape[2])
                hull[mask] = 0

        hull = rotate(hull, -angles[-1], mode='nearest', axes=(1,2), reshape=False)
        return hull.transpose(2, 0, 1)

def load_image(image_path):
        img = Image.open(image_path)
        return np.asarray(img)

def createBatch(ID, angles):
        X = []
        Y = []
        for i in range(batch_size):
                images = []
                pre_path = "data/"
                for angle in angles:
                        cur_path = pre_path + "2D/" + str(ID) + "/image_" + str(ID) + "_" + str(angle)+ "_LeftBrain_.png"
                        images.append(load_image(cur_path))

                hull = visualHull(images, angles)
                X.append(hull[...,np.newaxis])

                path_3D = pre_path + "3D/matrix3D_half_" + ID + "_LeftBrain_500.npy"
                img_3D = np.load(path_3D)/255
                img_3D = resize(img_3D, output_shape)[...,np.newaxis]
                Y.append(img_3D)

        return np.array(X), np.array(Y)

def generateData(IDs, n_images = None):
        while True:
                n_images = np.random.choice(np.arange(n_images_min, n_images_max+1)) if n_images == None else n_images
                id = np.random.choice(IDs, size=1)[0]
                angles = np.random.choice(180, n_images)
                yield createBatch(id, angles)


def splitData(split=0.8, n_images = None):
        ids = np.array(os.listdir(os.path.join('./data/2D/')))
        ids = ids[ids != 'BG001']
        np.random.shuffle(ids)
        split_len = int(len(ids) * split)

        training_ids = ids[:split_len]
        training = generateData(training_ids, n_images)

        validation_ids = ids[split_len:]
        validation = generateData(validation_ids, n_images)

        return training, validation

