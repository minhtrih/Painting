import os
import sys
import getopt
from os.path import join

PATH_TRAIN = ''  # PUT YOUR PATH HERE !!!

if __name__ == '__main__':
    PATH = 'data/original/'

    opts, args = getopt.getopt(sys.argv[1:], "sp:")
    for opt in opts:
        if '-p' in opt:
            command, PATH = opt
            if not os.path.exists(PATH):
                sys.exit('PATH NOT EXIST')

    path_images = []
    for path, sub_dirs, files in os.walk(PATH):
        for name in files:
            if name.split('.')[-1] == 'jpg':
                path_images.append(join(path, name.split('.')[0]))
    path_images.sort()

    path_labels = []
    for path, sub_dirs, files in os.walk(PATH):
        for name in files:
            if name.split('.')[-1] == 'txt' and name.split('.')[0] != 'classes':
                path_labels.append(join(path, name.split('.')[0]))
    path_labels.sort()

    num_images = len(path_images)
    num_labels = len(path_labels)
    if num_images != num_labels:
        sys.exit('Nums don\'t match')
    print('Lenghts: {} File JPG {} File TXT'.format(num_images, num_labels))

    for pi, pl in zip(path_images, path_labels):
        # print('Image {} - Labels {}'.format(pi, pl))
        if pi != pl:
            print(pi, pl)

    print("Saving ...")
    train = open('train.txt', 'w')
    for img in path_images:
        train.write(PATH_TRAIN + img + '.jpg\n')
    train.close()
