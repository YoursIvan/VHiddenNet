import bchlib
import random
import glob
import json
import os
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
BCH_POLYNOMIAL = 137
BCH_BITS = 7

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,default='100000')
    args = parser.parse_args()
    
    

    model = 'saved_models/ivan' + args.model
    images_dir = 'frames/'
    save_dir = 'frames_output/'
    secret_size = 5
    frames = 122
    files_list = []
    for i in range(frames):
        files_list.append(images_dir + str(i+1) + '.jpg')


    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    width = 480
    height = 360

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    stringset = []
    bit_set=[]
    for i in range(frames):
        stringset.append(''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',4)))
    
    #save string    

    count = 0
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        size = (width, height)
        print("Embedding...")
        for filename in files_list:
            count = count + 1
            image = Image.open(filename).convert('L')
            image = np.array(image,dtype=np.float32)
            image /= 255.
            image = np.expand_dims(image,-1)
            
            secret = stringset[count-1]
    
            data = bytearray(secret + ' '*(secret_size-len(secret)), 'utf-8')
            bit_set.append(''.join(format(x, '08b') for x in data))
            
            ecc = bch.encode(data)
            packet = data + ecc
        
            packet_binary = ''.join(format(x, '08b') for x in packet)
            secret = [int(x) for x in packet_binary]
            

            feed_dict = {input_secret:[secret],
                         input_image:[image]}

            hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

            rescaled = (hidden_img * 255).astype(np.uint8)
            raw_img = (image * 255).astype(np.uint8)
            residual = residual[0]+.5

            residual = (residual * 255).astype(np.uint8)

            save_name = filename.split('/')[-1].split('.')[0]

            im = Image.fromarray(np.squeeze(np.array(rescaled)))
            im.save(save_dir + '/'+ str(count) + '.jpg',quality = 65)

            #im = Image.fromarray(np.squeeze(np.array(residual)))
            #im.save(args.save_dir + '/residual'+args.stego_name,quality = 95)
            
    str_list = json.dumps(bit_set)
    a = open(r"message.txt", "w",encoding='UTF-8')
    a.write(str_list)
    a.close()
    print("Embedding Done.")

if __name__ == "__main__":
    main()
