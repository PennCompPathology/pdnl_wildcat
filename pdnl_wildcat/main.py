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
    args = parser.parse_args()

    params = json.load(open(args.parameters_file, 'r'))
    model = pdnl_wildcat.wrapper.Model(args.model_file, **params)

    frame = pdnl_sana.image.Frame(args.input_file)
    out = model.run(frame)

    class_names = params['class_names']
    arrs = {class_names[i]: out[i] for i in range(len(class_names))}
    np.savez(os.path.splitext(args.output_file)[0]+'.npz', **arrs)

if __name__ == "__main__":
    main()
