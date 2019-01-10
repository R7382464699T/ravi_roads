from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation, add, concatenate
from keras.optimizers import Adam
from keras import backend as K
smooth = 0.0001

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def roads_model(input_shape):
    concate = []
    inputs = Input(shape = input_shape)
    conv2d_1 = Conv2D(filters = 64,kernel_size = (3,3),padding = 'same', strides = (1,1))(inputs)
    conv2d_1 = BatchNormalization()(conv2d_1)
    conv2d_1 = Activation(activation='relu')(conv2d_1)
    conv2d_1 = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', strides = (1,1))(conv2d_1)
    
    conv2d_2 = Conv2D(filters = 64,kernel_size = (1,1),strides = (1,1))(inputs)
    conv2d_2 = BatchNormalization()(conv2d_2)
    
    conv2d_1 = add([conv2d_2,conv2d_1])
    concate.append(conv2d_1)
    
    conv2d_3 = BatchNormalization()(conv2d_1)
    conv2d_3 = Activation(activation='relu')(conv2d_3)
    conv2d_3 = Conv2D(filters = 128,kernel_size = (3,3),strides = (2,2),padding = 'same')(conv2d_3)
    conv2d_3 = BatchNormalization()(conv2d_3)
    conv2d_3 = Activation(activation='relu')(conv2d_3)
    conv2d_3 = Conv2D(filters = 128,kernel_size = (3,3),padding = 'same', strides = (1,1))(conv2d_3)
    
    conv2d_4 = Conv2D(filters = 128,kernel_size = (1,1), strides = (2,2))(conv2d_1)
    conv2d_4 = BatchNormalization()(conv2d_4)
    
    conv2d_3 = add([conv2d_4,conv2d_3])
    concate.append(conv2d_3)
    
    conv2d_5 = BatchNormalization()(conv2d_3)
    conv2d_5 = Activation(activation='relu')(conv2d_5)
    conv2d_5 = Conv2D(filters = 256,kernel_size = (3,3),padding = 'same', strides = (2,2))(conv2d_5)
    conv2d_5 = BatchNormalization()(conv2d_5)
    conv2d_5 = Activation(activation='relu')(conv2d_5)
    conv2d_5 = Conv2D(filters = 256,kernel_size = (3,3),padding = 'same', strides = (1,1))(conv2d_5)
    
    conv2d_6 = Conv2D(filters = 256,kernel_size = (1,1), strides = (2,2))(conv2d_3)
    conv2d_6 = BatchNormalization()(conv2d_6)
    
    conv2d_5 = add([conv2d_6,conv2d_5])
    concate.append(conv2d_5)
    
    conv2d_7 = BatchNormalization()(conv2d_5)
    conv2d_7 = Activation(activation='relu')(conv2d_7)
    conv2d_7 = Conv2D(filters = 512,kernel_size = (3,3),padding = 'same', strides = (2,2))(conv2d_7)
    conv2d_7 = BatchNormalization()(conv2d_7)
    conv2d_7 = Activation(activation='relu')(conv2d_7)
    conv2d_7 = Conv2D(filters = 512,kernel_size = (3,3),padding = 'same', strides = (1,1))(conv2d_7)
    
    conv2d_8 = Conv2D(filters = 512,kernel_size = (1,1), strides = (2,2))(conv2d_5)
    conv2d_8 = BatchNormalization()(conv2d_8)
    
    conv2d_7 = add([conv2d_8,conv2d_7])
    
    upsamp = UpSampling2D(size=(2,2))(conv2d_7)
    concate_1 = concatenate([upsamp,concate[2]],axis=3)
    
    conv2d_9 = BatchNormalization()(concate_1)
    conv2d_9 = Activation(activation = 'relu')(conv2d_9)
    conv2d_9 = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', strides = (1,1))(conv2d_9)
    conv2d_9 = BatchNormalization()(conv2d_9)
    conv2d_9 = Activation(activation = 'relu')(conv2d_9)
    conv2d_9 = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', strides = (1,1))(conv2d_9)
    
    conv2d_10 = Conv2D(filters = 256, kernel_size = (1,1), strides = (1,1))(concate_1)
    conv2d_10 = BatchNormalization()(conv2d_10)
    conv2d_10 = add([conv2d_9,conv2d_10])
    
    upsamp = UpSampling2D(size=(2,2))(conv2d_10)
    concate_2 = concatenate([upsamp,concate[1]],axis = 3)
    
    conv2d_11 = BatchNormalization()(concate_2)
    conv2d_11 = Activation(activation = 'relu')(conv2d_11)
    conv2d_11 = Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', strides = (1,1))(conv2d_11)
    conv2d_11 = BatchNormalization()(conv2d_11)
    conv2d_11 = Activation(activation = 'relu')(conv2d_11)
    conv2d_11 = Conv2D(filters = 128, kernel_size = (3,3),padding = 'same',strides = (1,1))(conv2d_11)
    
    conv2d_12 = Conv2D(filters = 128, kernel_size = (1,1), strides = (1,1))(concate_2)
    conv2d_12 = BatchNormalization()(conv2d_12)
    conv2d_12 = add([conv2d_11,conv2d_12])
    
    upsamp = UpSampling2D(size = (2,2))(conv2d_12)
    concate_3 = concatenate([upsamp,concate[0]],axis = 3)
    
    conv2d_13 = BatchNormalization()(concate_3)
    conv2d_13 = Activation(activation = 'relu')(conv2d_13)
    conv2d_13 = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', strides = (1,1))(conv2d_13)
    conv2d_13 = BatchNormalization()(conv2d_13)
    conv2d_13 = Activation(activation = 'relu')(conv2d_13)
    conv2d_13 = Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',strides = (1,1))(conv2d_13)
    
    conv2d_14 = Conv2D(filters = 64,kernel_size=(1,1),strides=(1,1))(concate_3)
    conv2d_14 = BatchNormalization()(conv2d_14)
    conv2d_14 = add([conv2d_13,conv2d_14])
    
    conv2d_final = Conv2D(filters = 1,kernel_size=(1,1),activation = 'sigmoid')(conv2d_14)
    
    model = Model(input=inputs, output=conv2d_final)
    model.compile(optimizer=Adam(lr=0.001), loss=dice_coef_loss, metrics=[dice_coef])
    return model
    


