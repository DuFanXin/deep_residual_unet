"""
https://arxiv.org/pdf/1711.10684.pdf
Road extraction by Deep Residual U-Net
"""

import datetime

from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, TensorBoard

from res_unet import *
from utils import *

# hyper parameters
model_name = "res_unet_"
input_shape = (512, 512, 1)
dataset_folder = ""
classes = []
batch_size = 2

model_file = model_name + datetime.datetime.today().strftime("_%d_%m_%y_%H:%M:%S") + ".h5"

model = build_res_unet(input_shape=input_shape)

model.summary()

optimizer = Adadelta()

model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

model_checkpoint = ModelCheckpoint(os.path.join("models", model_file), monitor='loss',
                                   save_best_only=True, verbose=True)
tensorboard = TensorBoard()

train_aug = ImageDataGenerator(vertical_flip=True, horizontal_flip=True)

train_gen = PASCALVOCIterator(directory=dataset_folder, target_file="train.txt",
                              image_data_generator=train_aug, target_size=(input_shape[0], input_shape[1]),
                              batch_size=batch_size, classes=classes)

model.fit_generator(train_gen, steps_per_epoch=300,
                    epochs=50,
                    callbacks=[tensorboard, model_checkpoint]
                    )

