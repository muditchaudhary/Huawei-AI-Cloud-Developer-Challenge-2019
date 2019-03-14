
#imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array
import numpy as np
import pickle

from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint



class Metrics(keras.callbacks.Callback):
    """
    Implementation of custom metrics: Precision, Recall, F-Measure and Confusion Matrix
    """
    
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        
        for i in range(len(self.validation_data)):
            x_val, y_val = self.validation_data.__getitem__(i)

        print(y_val.shape)
        print(y_val[1])
        print(y_val[2])

        y_predict = np.asarray(model.predict(self.validation_data, steps = 1))
        print(y_predict.shape)

        with tf.Session() as sess:
            lab = [i for i in range(0,75)]
            print(y_val.shape)

            y_val = np.argmax(y_val, axis=1)
            y_predict = np.argmax(y_predict, axis=1)
            print("y_val: ", y_val)
            print("y_predict:", y_predict)
            print("\nMetrics for Epoch")
            print("Confusion Matrix:\n",confusion_matrix(y_val,y_predict, labels=lab))
            print("Recall: ", recall_score(y_val,y_predict, average=None, labels=lab))
            print("Precision: ", precision_score(y_val,y_predict, average = None, labels=lab))
            print("F1_score: ", f1_score(y_val,y_predict, average =None, labels=lab))
            print("\n") 
            self._data.append({
                'val_recall': recall_score(y_val, y_predict, average = None, labels=lab),
                'val_precision': precision_score(y_val, y_predict, average = None, labels=lab),
                'val_f1_score': f1_score(y_val,y_predict, average = None, labels=lab),
            })
            return

    def get_data(self):
        return self._data
    
metrics = Metrics()



def lr_scheduler(epoch,lr):
    
    """
    Learning rate scheduler decays the learning rate by factor of 0.1 every 10 epochs after 20 epochs
    """
    decay_rate = 0.1
    if epoch==10:
        return lr*decay_rate
    elif epoch%10==0 and epoch >20:
        return lr*decay_rate
    return lr

# To put in the model, put it inside callbacks
LRScheduler = keras.callbacks.LearningRateScheduler(lr_scheduler,verbose=1)



#Data Generator for Data augmentation. Edit possible
train_datagen = ImageDataGenerator(
	rotation_range=40,
	width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
	validation_split = 0.008)
#333 validation






train_generator = train_datagen.flow_from_directory(
        'datasets/large_sample',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=28,
        class_mode='categorical',
        subset = 'training')  # since we use binary_crossentropy loss, we need binary labels



val_generator = train_datagen.flow_from_directory(
        'datasets/large_sample',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=28,
        class_mode='categorical',
        subset = 'validation')  # since we use binary_crossentropy loss, we need binary labels



labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
print(labels)



#Naive Model
input_tensor = keras.layers.Input(shape = (150,150,3))
model = ResNet50(input_tensor=input_tensor, include_top=False, weights = None)
x=model.output

x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(x) 
x = tf.keras.layers.Dense(1024,activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x=tf.keras.layers.Dense(75,activation ='softmax')(x)
model = tf.keras.models.Model(model.input, x)


model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=0.01)
              ,metrics=['accuracy'])


checkpoint = ModelCheckpoint('model_resA.h5',monitor='val_acc',save_best_only=True,period=1,verbose=0)
checkpointB = ModelCheckpoint('model_resB.h5', period = 1,verbose=0)
print(model.summary())

#model.load_weights('weights_resA.h5')
#history=model.fit_generator(train_generator,
#                   steps_per_epoch = 1800,
#                   epochs = 20, validation_data = val_generator, callbacks=[checkpoint,LRScheduler])
#file_pi = open('Train_histA',"wb")
#pickle.dump(history.history,file_pi)
#file_pi.close()

#model = load_model('model_ResA.h5')
model=load_model('model_resA.h5')
img = load_img('datasets/large_sample/almond_desert/6350.jpg')  # this is a PIL image
img=img.resize((150,150))
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

print(x.shape)


y = model.predict(x)

print(y)

print(labels[np.argmax(y)])


for i,layer in enumerate(model.layers):
  print(i,layer.name)



#Data Generator for Data augmentation. Edit possible
val_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split = 0.2)



small_train_generator = val_datagen.flow_from_directory(
        'datasets/small_sample',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='categorical',
        subset = 'training')



small_val_generator = val_datagen.flow_from_directory(
        'datasets/small_sample',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='categorical',
        subset = 'validation')


for layer in model.layers[:-4]:
    layer.trainable=False



for i,layer in enumerate(model.layers):
  print(i,layer.trainable)



model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
new_model = tf.keras.models.Model(model.input, model.layers[-5].output)

print(new_model.summary())

y=new_model.output

y = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(y) 
y = tf.keras.layers.Dense(1024,activation='relu')(y)
y = tf.keras.layers.Dropout(0.4)(y)
y=tf.keras.layers.Dense(25,activation ='softmax')(y)

new_model = tf.keras.models.Model(new_model.input, y)

new_model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])



#new_model.summary()



for i,layer in enumerate(model.layers):
  print(i,layer.trainable)

new_model = load_model('model_resB.h5')

#historyB=new_model.fit_generator(small_train_generator,
#                   steps_per_epoch =7,
 #                  epochs = 10, validation_data = small_val_generator,callbacks=[checkpointB])
##file_pi = open('Train_histB',"wb")
#pickle.dump(historyB.history, file_pi)
#file_pi.close()


img = load_img('datasets/small_sample/stir_fried_lettuce/6581.jpg')  # this is a PIL image
img=img.resize((150,150))
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

print(x.shape)



labelsSmall = (small_train_generator.class_indices)
labelsSmall = dict((v,k) for k,v in labelsSmall.items())
print(labelsSmall, end="\n")
y = new_model.predict(x)


print(y)
print(labelsSmall[np.argmax(y)])

