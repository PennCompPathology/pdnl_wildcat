#!/usr/bin/env python3

import os
import sys
import argparse
import json

from tqdm import tqdm
import numpy as np
import pdnl_sana.image

import pdnl_wildcat.wrapper
import pdnl_sana.logging

from matplotlib import pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', help="path to image data", required=True)
    parser.add_argument('-o', '--output_path', help=".npy path to write outputs to", required=True)
    parser.add_argument('-m', '--model_file', help="path to .json file defining the wildcat model", required=True)

    args = parser.parse_args()

    params = json.load(open(args.model_file, 'r'))
    model = pdnl_wildcat.wrapper.Model(**params)
    model_name = os.path.splitext(os.path.basename(args.model_file))[0]

    os.makedirs(args.output_path, exist_ok=True)
    
    print('Deploying model to chunks...', flush=True)
    for d in tqdm(os.listdir(args.input_path)):
        if not os.path.isdir(os.path.join(args.input_path, d)):
            continue
        
        i, j = list(map(int, d.split('_')))

        in_d = os.path.join(args.input_path, f'{i}_{j}')
        out_d = os.path.join(args.output_path, f'{i}_{j}')
        os.makedirs(out_d, exist_ok=True)

        logger = pdnl_sana.logging.Logger('normal', os.path.join(in_d, 'log.pkl'))

        level = logger.data['level']
        mpp = logger.data['mpp']
        ds = logger.data['ds']
        converter = pdnl_sana.geo.Converter(mpp, ds)
        frame = pdnl_sana.image.Frame(os.path.join(in_d, 'frame.png'), level=level, converter=converter)
        mask = pdnl_sana.image.Frame(os.path.join(in_d, 'mask.png'), level=level, converter=converter)        
    
        out = model.run(frame)

        class_names = params['class_names']
        arrs = {class_names[i]: out[i] for i in range(len(class_names))}
        
        np.savez(os.path.join(out_d, f'wildcat_proba.npz'), **arrs)

if __name__ == "__main__":
    main()
