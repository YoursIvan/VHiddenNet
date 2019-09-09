import numpy as np
import os
import lpips.lpips_tf as lpips_tf
import tensorflow as tf
import utils
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import regularizers
from stn import spatial_transformer_network as stn_transformer

normalize_coefficient = 0.002

#768x576#

class StegaStampEncoder(Layer):
    def __init__(self, height, width):
        super(StegaStampEncoder, self).__init__()
        self.secret_dense = Dense(2700, activation='relu', kernel_initializer='he_normal')
        self.secret_conv =  Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        '''
        self.conv1 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv4 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv5 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv6 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv7 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv8 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv9 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv10 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv11 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv12 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv13 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv14 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv15 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv16 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv17 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.conv18 = Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        self.residual = Conv2D(1, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal')
        '''
        self.conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        self.conv3 = Conv2D(128, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        self.conv4 = Conv2D(256, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        #self.conv5 = Conv2D(256, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        
        #self.up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        #self.conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        self.up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        self.conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        self.up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        self.conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        self.up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        self.conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        self.conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        self.residual = Conv2D(1, 1, activation=None, padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        
        

    def call(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.secret_dense(secret)
        secret = Reshape((45, 60, 1))(secret)
        secret_enlarged = UpSampling2D(size=(8,8))(secret)
        
        secret_channel = self.secret_conv(secret_enlarged)

        inputs = concatenate([secret_channel, image], axis=-1)
        
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        '''
        conv5 = self.conv5(conv4)
        up6 = self.up6(UpSampling2D(size=(2,2))(conv5))
        merge6 = concatenate([conv4,up6], axis=3)
        conv6 = self.conv6(merge6)
        '''
        up7 = self.up7(UpSampling2D(size=(2,2))(conv4))
        merge7 = concatenate([conv3,up7], axis=3)
        conv7 = self.conv7(merge7)
        up8 = self.up8(UpSampling2D(size=(2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis=3)
        conv8 = self.conv8(merge8)
        up9 = self.up9(UpSampling2D(size=(2,2))(conv8))
        merge9 = concatenate([conv1,up9,inputs], axis=3)
        conv9 = self.conv9(merge9)
        conva = self.conv9(merge9)
        conv10 = self.conv10(conv9)
        residual = self.residual(conv9)
        '''
        conv = self.conv1(inputs)
        conv = self.conv2(conv)
        conv = self.conv3(conv)
        conv = self.conv4(conv)
        conv = self.conv5(conv)
        conv = self.conv6(conv)
        conv = self.conv7(conv)
        conv = self.conv8(conv)
        conv = self.conv9(conv)
        conv = self.conv10(conv)
        conv = self.conv11(conv)
        conv = self.conv12(conv)
        conv = self.conv13(conv)
        conv = self.conv14(conv)
        conv = self.conv15(conv)
        conv = self.conv16(conv)
        conv = self.conv17(conv)
        conv = self.conv18(conv)
        residual = self.residual(conv)
        '''
        return residual

class StegaStampDecoder(Layer):
    def __init__(self, secret_size, height, width):
        super(StegaStampDecoder, self).__init__()
        self.height = height
        self.width = width
        
        self.VDSR = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            
            Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(normalize_coefficient))
        ])

        self.decoder = Sequential([
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(32, (3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Flatten(),
            Dense(1024,activation='relu'),
            Dense(secret_size)
        ])

    def call(self, image):
        image = image - .5
        #res_img = self.VDSR(image)
        #HD_img = tf.add(res_img, image)
        return self.decoder(image)

class Discriminator(Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = Sequential([
            Conv2D(8, (3, 3), strides=2, activation='relu', padding='same',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(16, (3, 3), strides=2, activation='relu', padding='same',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same',kernel_regularizer=regularizers.l2(normalize_coefficient)),
            Conv2D(1, (3, 3), activation=None, padding='same')
        ])

    def call(self, image):
            x = image - .5
            x = self.model(x)
            output = tf.reduce_mean(x)
            return output,x

def transform_net(encoded_image, args, global_step):
    sh = tf.shape(encoded_image)
    ramp_fn = lambda ramp : tf.minimum(tf.to_float(global_step) / ramp, 1.)


    jpeg_quality = args.jpeg_quality
    jpeg_factor = tf.cond(tf.less(jpeg_quality, 50), lambda: 5000. / jpeg_quality, lambda: 200. - jpeg_quality * 2) / 100. + .0001


    encoded_image = tf.reshape(encoded_image, [-1,360,480,1])
    if not args.no_jpeg:
        encoded_image = utils.jpeg_compress_decompress_y(encoded_image, rounding=utils.round_only_at_0, factor=jpeg_factor, downsample_c=True)
    
    summaries = [tf.summary.scalar('transformer/jpeg_quality', jpeg_quality)]
    return encoded_image, summaries


def interframes_loss(residual,batch_size):
    inter_loss = 0
    for i in range(1,batch_size-1):
        inter_loss += tf.reduce_mean(tf.abs(tf.subtract(residual[i,:,:], residual[i-1,:,:])) + tf.abs(tf.subtract(residual[i+1,:,:], residual[i,:,:])),axis = [0,1,2])
    return inter_loss/(batch_size-2)


def get_secret_acc(secret_true,secret_pred):
    with tf.variable_scope("acc"):
        secret_pred = tf.round(secret_pred)
        correct_pred = tf.count_nonzero(secret_pred - secret_true, axis=1)

        str_acc = 1.0 - tf.count_nonzero(correct_pred - tf.to_int64(tf.shape(secret_pred)[1])) / tf.size(correct_pred, out_type=tf.int64)

        bit_acc = tf.reduce_sum(correct_pred) / tf.size(secret_pred, out_type=tf.int64)
        return bit_acc, str_acc


def build_model(encoder,
                decoder,
                discriminator,
                secret_input,
                image_input,
                l2_edge_gain,
                borders,
                secret_size,
                M,
                loss_scales,
                yuv_scales,
                args,
                global_step):

    input_warped = tf.contrib.image.transform(image_input, M[:,1,:], interpolation='BILINEAR')
    mask_warped = tf.contrib.image.transform(tf.ones_like(input_warped), M[:,1,:], interpolation='BILINEAR')
    input_warped += (1-mask_warped) * image_input

    
    residual_warped = encoder((secret_input, input_warped))
    encoded_warped = residual_warped + input_warped
    residual = tf.contrib.image.transform(residual_warped, M[:,0,:], interpolation='BILINEAR')
    


    if borders == 'no_edge':
        encoded_image = image_input + residual
    elif borders == 'black':
        encoded_image = residual_warped + input_warped
        encoded_image = tf.contrib.image.transform(encoded_image, M[:,0,:], interpolation='BILINEAR')
        input_unwarped = tf.contrib.image.transform(input_warped, M[:,0,:], interpolation='BILINEAR')
    elif borders.startswith('random'):
        mask = tf.contrib.image.transform(tf.ones_like(residual), M[:,0,:], interpolation='BILINEAR')
        encoded_image = residual_warped + input_warped
        encoded_image = tf.contrib.image.transform(encoded_image, M[:,0,:], interpolation='BILINEAR')
        input_unwarped = tf.contrib.image.transform(input_warped, M[:,0,:], interpolation='BILINEAR')
        ch = 3 if borders.endswith('rgb') else 1
        encoded_image += (1-mask) * tf.ones_like(residual) * tf.random_uniform([ch])
    elif borders == 'white':
        mask = tf.contrib.image.transform(tf.ones_like(residual), M[:,0,:], interpolation='BILINEAR')
        encoded_image = residual_warped + input_warped
        encoded_image = tf.contrib.image.transform(encoded_image, M[:,0,:], interpolation='BILINEAR')
        input_unwarped = tf.contrib.image.transform(input_warped, M[:,0,:], interpolation='BILINEAR')
        encoded_image += (1-mask) * tf.ones_like(residual)
    elif borders == 'image':
        mask = tf.contrib.image.transform(tf.ones_like(residual), M[:,0,:], interpolation='BILINEAR')
        encoded_image = residual_warped + input_warped
        encoded_image = tf.contrib.image.transform(encoded_image, M[:,0,:], interpolation='BILINEAR')
        encoded_image += (1-mask) * tf.manip.roll(image_input, shift=1, axis=0)

    if borders == 'no_edge':
        D_output_real, _= discriminator(image_input)
        D_output_fake, D_heatmap  = discriminator(encoded_image)
    else:
        D_output_real, _ = discriminator(input_warped)
        D_output_fake, D_heatmap = discriminator(encoded_warped)

    transformed_image, transform_summaries = transform_net(encoded_image, args, global_step)

    decoded_secret = decoder(transformed_image)

    bit_acc, str_acc = get_secret_acc(secret_input, decoded_secret)

    lpips_loss_op = tf.reduce_mean(lpips_tf.lpips(tf.tile(image_input,[1,1,1,3]), tf.tile(encoded_image,[1,1,1,3])))
    secret_loss_op = tf.losses.sigmoid_cross_entropy(secret_input, decoded_secret)

    size = (int(image_input.shape[1]),int(image_input.shape[2]))
    gain = 10
    falloff_speed = 4 # Cos dropoff that reaches 0 at distance 1/x into image
    falloff_im = np.ones(size)
    for i in range(int(falloff_im.shape[0]/falloff_speed)):
        falloff_im[-i,:] *= (np.cos(4*np.pi*i/size[0]+np.pi)+1)/2
        falloff_im[i,:] *= (np.cos(4*np.pi*i/size[0]+np.pi)+1)/2
    for j in range(int(falloff_im.shape[1]/falloff_speed)):
        falloff_im[:,-j] *= (np.cos(4*np.pi*j/size[0]+np.pi)+1)/2
        falloff_im[:,j] *= (np.cos(4*np.pi*j/size[0]+np.pi)+1)/2
    falloff_im = 1-falloff_im
    falloff_im = tf.convert_to_tensor(falloff_im, dtype=tf.float32)
    falloff_im *= l2_edge_gain

    encoded_image_yuv = encoded_image
    image_input_yuv = image_input
    im_diff = encoded_image_yuv-image_input_yuv
    im_diff += im_diff * tf.expand_dims(falloff_im, axis=[-1])
    yuv_loss_op = tf.reduce_mean(tf.square(im_diff), axis=[0,1,2,3])
    image_loss_op = yuv_loss_op

    D_loss = D_output_real - D_output_fake
    G_loss = D_output_fake
    inter_loss_op = interframes_loss(residual,args.batch_size)
    
    #loss_op =  loss_scales[1]*lpips_loss_op + loss_scales[2]*secret_loss_op + inter_loss
    loss_op = loss_scales[0]*image_loss_op + loss_scales[1]*lpips_loss_op + loss_scales[2]*secret_loss_op 
    
    if not args.no_gan:
       loss_op += loss_scales[3]*G_loss
       
    loss_inter_op = loss_op + 10*inter_loss_op

    summary_op = tf.summary.merge([
        tf.summary.scalar('bit_acc', bit_acc, family='train'),
        tf.summary.scalar('str_acc', str_acc, family='train'),
        tf.summary.scalar('loss', loss_op, family='train'),
        tf.summary.scalar('image_loss', image_loss_op, family='train'),
        tf.summary.scalar('lpip_loss', lpips_loss_op, family='train'),
        tf.summary.scalar('G_loss', G_loss, family='train'),
        tf.summary.scalar('secret_loss', secret_loss_op, family='train'),
        tf.summary.scalar('dis_loss', D_loss, family='train'),
        tf.summary.scalar('Y_loss', yuv_loss_op, family='color_loss'),
        #tf.summary.scalar('U_loss', yuv_loss_op[1], family='color_loss'),
        #tf.summary.scalar('V_loss', yuv_loss_op[2], family='color_loss'),
        tf.summary.scalar('inter_loss_op', inter_loss_op, family='interframe_loss'),
    ] + transform_summaries)

    image_summary_op = tf.summary.merge([
        image_to_summary(image_input, 'image_input', family='input'),
        image_to_summary(input_warped, 'image_warped', family='input'),
        image_to_summary(encoded_warped, 'encoded_warped', family='encoded'),
        image_to_summary(residual_warped+.5, 'residual', family='encoded'),
        image_to_summary(encoded_image, 'encoded_image', family='encoded'),
        image_to_summary(transformed_image, 'transformed_image', family='transformed'),
        image_to_summary(D_heatmap, 'discriminator', family='losses'),
    ])

    return loss_op, secret_loss_op, D_loss,G_loss,image_loss_op,lpips_loss_op ,summary_op, image_summary_op, bit_acc,inter_loss_op,loss_inter_op

def image_to_summary(image, name, family='train'):
    image = tf.clip_by_value(image, 0, 1)
    image = tf.cast(image * 255, dtype=tf.uint8)
    summary = tf.summary.image(name,image,max_outputs=1,family=family)
    return summary

def prepare_deployment_hiding_graph(encoder, secret_input, image_input):

    residual = encoder((secret_input, image_input))
    encoded_image = residual + image_input
    encoded_image = tf.clip_by_value(encoded_image, 0, 1)

    return encoded_image, residual

def prepare_deployment_reveal_graph(decoder, image_input):
    decoded_secret = decoder(image_input)

    return tf.round(tf.sigmoid(decoded_secret))