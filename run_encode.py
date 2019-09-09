import os
import random
frames = 192

stringset = []
for i in range(frames):
    stringset.append(''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',25)))


for i in range(frames):
    print(stringset[i])
    command = '/cyf/Python3.6.7/bin/python3.6 encode_image.py saved_models/ivan40000  --image frames/' + str(i+1) + '.jpg --save_dir frames_output/ --stego_name ' +  str(i+1) + '.jpg --secret ' + stringset[i]
    os.system(command)