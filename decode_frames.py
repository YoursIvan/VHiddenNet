import bchlib
import os
import glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import json

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
    images_dir = 'frames_output/'
    secret_size = 5
    frames = 122
    files_list = []
    for i in range(frames):
        files_list.append(images_dir + str(i+1) + '.jpg')


    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    ext_message = []
    
    print("Extracting...")
    for filename in files_list:
        image = Image.open(filename).convert('L')
        image = np.array(image,dtype=np.float32)
        image /= 255.
        image = np.expand_dims(image,-1)

        feed_dict = {input_image:[image]}

        secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]

        packet_binary = "".join([str(int(bit)) for bit in secret])
        packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
        bitflips = bch.decode_inplace(data, ecc)
        ext_message.append(''.join(format(x, '08b') for x in data))
    
    
    print("Extracting Done.")    
    str_list = json.dumps(ext_message)
    a = open(r"message_ext.txt", "w",encoding='UTF-8')
    a.write(str_list)
    a.close()
    

    b = open(r"message.txt", "r",encoding='UTF-8')
    message = b.read()
    message = json.loads(message)
    
    ber = 0
    error_char = 0
    for cover,stego in zip(message,ext_message):
        for c1,c2 in zip(cover,stego):
          if c1 != c2:
              error_char+=1
    print("Error_Char_Count:" + str(error_char))
    ber = error_char/(len(message)*6*8)
    print("BER:" + str(ber))


if __name__ == "__main__":
    main()
  
