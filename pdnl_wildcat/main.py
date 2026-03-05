#!/usr/bin/env python3

import os
import sys
import argparse
import json

import numpy as np
import pdnl_sana.image

import pdnl_wildcat.wrapper

from matplotlib import pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help="path to image file to process", required=True)
    parser.add_argument('-o', '--output_file', help=".npy path to write outputs to", required=True)
    parser.add_argument('-m', '--model_file', help="path to .dat file which stores model weights", required=True)
    parser.add_argument('-p', '--parameters_file', help="path to .json file which stores model parameters", required=True)
    parser.add_argument('--save_proba', action='store_true', help="if given, stores the raw probability outputs")
    parser.add_argument('--debug', action='store_true', help="plots an output image to analyze")
    args = parser.parse_args()

    params = json.load(open(args.parameters_file, 'r'))
    model = pdnl_wildcat.wrapper.Model(args.model_file, **params)

    frame = pdnl_sana.image.Frame(args.input_file)
    out = model.run(frame)

    
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0)
    proba = softmax(out)
    
    if args.save_proba:
        np.save(os.path.splitext(args.output_file)[0]+'_proba.npy', proba)

    pred = np.argmax(proba, axis=0)
    np.save(os.path.splitext(args.output_file)[0]+'.npy', pred)

    if args.debug:
        class_names = params['class_names']
        fig, axs = plt.subplots(1,1+len(class_names)+1, sharex=True, sharey=True)
        axs[0].imshow(frame.img)
        axs[0].set_title('Original')
        for i in range(len(class_names)):
            axs[1+i].imshow(proba[i], vmin=0, vmax=1)
            axs[1+i].set_title(class_names[i])
        axs[-1].imshow(pred, vmin=0, vmax=proba.shape[0])
        axs[-1].set_title('Predictions')
        plt.show()
    
if __name__ == "__main__":
    main()
