import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

import source.neuralnet as nn
import source.datamanager as dman
import source.tf_process as tfp
import cv2
import numpy as np
DE_HEIGHT = 512

def main():

    srnet = nn.SRNET()

    dataset = dman.DataSet()

    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    
    # tfp.training(sess=sess, neuralnet=srnet, saver=saver, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch)
    # tfp.validation(sess=sess, neuralnet=srnet, saver=saver, dataset=dataset)
    # tfp.test(img_path, sess=sess, neuralnet=srnet, saver=saver, dataset=dataset)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        w, h, c = np.shape(frame)
        frame = cv2.resize(frame, (DE_HEIGHT, (int)(w * DE_HEIGHT / h)))
        frame0 = cv2.blur(frame, (4, 4))
        cv2.imshow('origin', frame0)
        key = cv2.waitKey(3)

        if key == 32 or key == 13:
            tfp.test(frame, sess=sess, neuralnet=srnet, saver=saver, dataset=dataset)
        if key == 27:
            print('Stopping video by ESC')
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5000, help='-')
    parser.add_argument('--batch', type=int, default=16, help='-')

    FLAGS, unparsed = parser.parse_known_args()

    main()
