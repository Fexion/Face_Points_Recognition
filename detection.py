from keras.models import Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers import Input

import numpy as np

from os.path import join
from skimage.transform import resize
from scipy.misc import imread
from os import listdir



def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = np.array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res


def train_detector(train_gt, train_img_dir, fast_train = False):#dict train_gt, string train_img_dir
    train_shape = 128
    shape = (train_shape, train_shape, 3)
    filenames = listdir(train_img_dir)
    num_files = len(filenames)
    X_train = np.zeros((num_files,train_shape,train_shape,3), dtype = "float64")
    Y_train = np.zeros((num_files, 28), dtype = "float64")


    for i, filename in enumerate(filenames):
        y = np.zeros(28, dtype = "float64")
        img = imread(join(train_img_dir, filename), mode = "RGB") #grayscale might work

        rows, col = img.shape[0], img.shape[1]
        X_train[i] = resize(img, (train_shape, train_shape))
        del img
        for j in range(14):
            y[j*2] = (train_gt[filename][j*2] * train_shape)// col
            y[j*2 + 1] = (train_gt[filename][j*2 + 1] * train_shape)// rows
        Y_train[i] = y


    num_train= num_files
    num_classes = 28



    mean_num = X_train.mean(axis = (0,1,2))
    X_train -= mean_num

    if fast_train:
        num_epochs = 1
    else:
        num_epochs = 55


    batch_size = 16
    filter_num0 = 8
    filter_num1 = 16
    filter_num2 = 32
    filter_num3 = 64
    filter_num4 = 128
    ks = (5,5)#(3,3)
    stride = (1,1)#(1,1)

    inp = Input(shape)

    conv0 = Conv2D(filters=filter_num0, kernel_size=ks, strides = stride, padding="same",
                    activation="relu")(inp)
    pool0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv0)

    conv1 = Conv2D(filters=filter_num1, kernel_size=ks, strides = stride, padding="same",
                    activation="relu")(pool0)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv1)


    conv2 = Conv2D(filters=filter_num2, kernel_size=ks, strides = stride, padding="same",
                    activation="relu")(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding="same")(conv2)


    conv3 = Conv2D(filters=filter_num3, kernel_size=ks, strides = stride, padding="same",
                    activation="relu")(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv3)


    conv4 = Conv2D(filters=filter_num4, kernel_size=ks, strides = stride, padding="same",
                    activation="relu")(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv4)


    flat = Flatten()(pool4)
    dense = Dense(512, activation="relu")(flat)
    out = Dense(num_classes)(dense)

    model = Model(inp, out)


    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, Y_train,
              batch_size=batch_size, epochs=num_epochs,
              verbose=1)
    return model

def detect(model, test_img_dir):
    test_shape = 128

    filenames = listdir(test_img_dir)
    X = np.zeros((len(filenames),test_shape,test_shape,3), dtype = "float64")
    rows_col = np.zeros((len(filenames),2), dtype = "float64")
    for i, filename in enumerate(filenames):
        img = imread(join(test_img_dir, filename), mode = "RGB") #grayscale might work

        rows_col[i] =  img.shape[0:2:]
        X[i] = resize(img, (test_shape, test_shape))
        del img

    mean_num = X.mean(axis = (0,1,2))
    X-=mean_num
    prediction = model.predict(X)

    result = {}
    for i, filename in enumerate(filenames):
        rows, col = rows_col[i]
        for j in range(14):
            prediction[i][j*2] *= (col/test_shape)
            prediction[i][j*2 + 1] *= (rows/test_shape)
        result[filename] = np.rint(prediction[i])
    return result


def dots_to_img(test_img_dir, dots):
    for i, filename in enumerate(dots.keys()):
        #print(prediction[0], "\n", np.max(result), np.min(result))
        img = imread(join(test_img_dir, filename), mode = "RGB")
        for j in range(14):
            #print(np.max(img))
            img = resize(img, img.shape[:2])
            img[dots[filename][j*2+1]][dots[filename][j*2]] = 1.5


            res_img = image.array_to_img(img)

            res_img.save("saves/" + str(i) + ".jpg")


if __name__ == "__main__":
    train_dir = "img"

    train_gt = read_csv(join(train_dir, 'gt.csv'))
    train_img_dir = join(train_dir, 'images')

    model = train_detector(train_gt, train_img_dir, fast_train=False)
    model.save('facepoints_model_new.hdf5')
    model = load_model("facepoints_model_new.hdf5")
    dots = detect(model, "test_img")
    dots_to_img("test_img", dots)




















1
