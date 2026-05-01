
# system modules
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# installed modules
import torch
from torchvision import transforms
import scipy
import numpy as np
from PIL import Image
from tqdm import tqdm

# custom modules
import pdnl_sana.image
from pdnl_wildcat.unet_wildcat import resnet50_wildcat_upsample

class Model:
    def __init__(self, model_path, num_classes, mpp=0.5045, kmax=0.02, alpha=0.7, num_maps=4, kmin=0.0, class_names=[], debug=None):
        self.model_path = model_path
        self.num_classes = num_classes
        self.debug = debug

        # grab the device to run the model on
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # load the model
        self.model = resnet50_wildcat_upsample(self.num_classes, pretrained=False,
            kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self.model = self.model.to(self.device)

        # resolution that was used to train the model
        self.mpp = mpp
        
        # size of patch to be inputted to the model
        self.patch_raw = 448 # default: 112

        # padding, relative to patch_size to add to the window
        self.padding_rel = 1/7 # default: 1.0
        self.padding_raw = int(self.padding_rel * self.patch_raw) # 56

        # amount wildcat shrinks input images when mapping to segmentations
        self.wildcat_ds = 2.0

        # padding size for the output
        self.padding_out = int(self.padding_rel * self.patch_raw / self.wildcat_ds)
    #
    # end of constructor

    def run(self, frame, debug=False, get_coords=False, deploy_grid=True):
        self.frame = frame

        # resize the frame to the target mpp
        ds = self.mpp / self.frame.converter.mpp
        orig_size = self.frame.size()
        new_size = orig_size / ds
        self.frame.resize(new_size)

        # size of image to process
        self.image_size = np.array(self.frame.img.shape[:2])

        # true final patch size to output
        self.true_patch_size = self.patch_raw-self.padding_raw

        # dimension of the final output image
        self.out_dim = self.true_patch_size*np.ceil(self.image_size/self.true_patch_size).astype(int)

        # initialize the output array
        output = np.zeros((self.num_classes, self.out_dim[0], self.out_dim[1]))
        
        coords = []
        if deploy_grid:
            # loop over windows
            for v in range(0, self.image_size[0], self.patch_raw-self.padding_raw): #height
                for u in range(0, self.image_size[1], self.patch_raw-self.padding_raw): #width

                    # subtract the padding
                    x0, y0 = u - (self.padding_raw//2), v - (self.padding_raw//2)
                    x1, y1 = x0 + self.patch_raw, y0 + self.patch_raw

                    xpad0, xpad1, ypad0, ypad1 = 0,0,0,0
                    if x0 < 0:
                        xpad0 = 0 - x0
                        x0 = 0
                    if y0 < 0:
                        ypad0 = 0 - y0
                        y0 = 0
                    if x1 > self.image_size[1]:
                        xpad1 = x1 - self.image_size[1]
                        x1 = self.image_size[1]
                    if y1 > self.image_size[0]:
                        ypad1 = y1 - self.image_size[0]
                        y1 = self.image_size[0]
                    
                    coords.append([(x0,y0),(x1,y1)])
                    chunk = self.frame.img[y0:y1,x0:x1]
                    chunk = np.pad(chunk, [(ypad0,ypad1),(xpad0,xpad1),(0,0)], mode='constant', constant_values=255)
                    chunk = Image.fromarray(chunk)

                    # # compute the desired size of input
                    # # TODO: input_size should be here as a ratio
                    # wwc = int(wp * self.patch_raw / self.patch_raw)

                    # resample the chunk for the two networks
                    tran = transforms.Compose([
                        # transforms.Resize((wwc, wwc)),
                        transforms.ToTensor(),
                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                    # convert the read chunk to tensor format
                    with torch.no_grad():

                        # Apply transforms and turn into correct size torch tensor
                        chunk_tensor = torch.unsqueeze(tran(chunk), dim=0).to(self.device)

                        # forward pass through the model
                        x_clas = self.model.forward_to_classifier(chunk_tensor)
                        x_cpool = self.model.spatial_pooling.class_wise(x_clas)

                        # scale the image to desired size
                        x_cpool_up = torch.nn.functional.interpolate(x_cpool, scale_factor=self.wildcat_ds)

                        # extract the central portion of the output
                        p0 = 0 + (self.padding_raw//2)
                        p1 = self.patch_raw - (self.padding_raw//2)
                        x_cpool_ctr = x_cpool_up[:,:,p0:p1,p0:p1]

                        # place in the output
                        xout0, xout1 = u, u+self.true_patch_size
                        yout0, yout1 = v, v+self.true_patch_size

                        output[:, yout0:yout1, xout0:xout1] = x_cpool_ctr[0,:,:,:].cpu().detach().numpy()
                    #
                    # end of model evaluation
            #
            # end of window loop
            
            output = output[:,:self.image_size[0],:self.image_size[1]]

        else:
            frame = Image.fromarray(self.frame.img)
            # resample the chunk for the two networks
            tran = transforms.Compose([
                # transforms.Resize((wwc, wwc)),
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            frame_tensor = torch.unsqueeze(tran(frame), dim=0).to(self.device)

            x_clas = self.model.forward_to_classifier(frame_tensor)
            x_cpool = self.model.spatial_pooling.class_wise(x_clas)
            x_cpool_up = torch.nn.functional.interpolate(x_cpool, scale_factor=self.wildcat_ds) #shape: [1,3,596,596]
            output = x_cpool_up[0,:,:,:].cpu().detach().numpy()

        # resize the output to the original resolution
        frame = pdnl_sana.image.Frame(output.transpose(1,2,0), level=self.frame.level, converter=self.frame.converter)
        frame.resize(orig_size)
        output = frame.img.transpose(2,0,1)
            
        if get_coords and deploy_grid:
            return output, coords
        else:
            return output
    #
    # end of run
#
# end of Model
