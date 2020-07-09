# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
import cv2
import time

import rospy
from std_msgs.msg import String


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--video_path', type=str,
                        help='path to a test video', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono+stereo_320x192",
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    vid = cv2.VideoCapture(args.video_path)
    
    # FINDING INPUT IMAGES
    if os.path.isfile(args.video_path):
        # Only testing on a single image
        paths = [args.video_path]
        output_directory = os.path.dirname(args.video_path)
    elif os.path.isdir(args.video_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.video_path, '*.{}'.format(args.ext)))
        output_directory = args.video_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.video_path))
    

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        while(True):
            prev_time = time.time()

            ret, frame = vid.read()
            cv2.imshow("original",frame)
            dst = frame.copy()
            dst_height = dst.shape[0]
            dst_width = dst.shape[1]
            dst_mid_width = int(dst_width)/2
            dst = frame[(int(dst_height)-320):int(dst_height), (int(dst_mid_width)-320): (int(dst_mid_width)+320)]
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            input_image = pil.fromarray(dst)
            
            # Load image and preprocess
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            #output_name = os.path.splitext(os.path.basename(args.video_path))[0]
            #name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            #scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            #np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            
            #curr_time = time.time()
            #exec_time = curr_time - prev_time

            imcv = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)            
            #info = "time:" + str(round(1000*exec_time, 2)) + " ms, FPS: " + str(round((1000/(1000*exec_time)),1))
            #cv2.putText(imcv, text=info, org=(50, 70),
            #    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #    fontScale=1, color=(255, 0, 0), thickness=2)
            #cv2.imshow("result",imcv)
            imcv_gray = cv2.cvtColor(imcv, cv2.COLOR_BGR2GRAY)
            imcv_edges = cv2.Canny(imcv_gray, 50, 100)
            imcv_height = imcv_edges.shape[0]
            imcv_width = imcv_edges.shape[1]
            imcv_min = 0
            imcv_max = int(imcv_width)
            object_width = imcv_max - imcv_min
            object_height = 180
            while (imcv_max - imcv_min) >= 300:
                for x in range(0,imcv_width):
                    if imcv_edges.item(object_height, x) == 255:    # y,x, ch
                        if x <= int(imcv_width) / 2:
                            imcv_min = x
                        else:
                            imcv_max = x
                            break;
                if object_height == 0:
                    break;
                else:
                    object_height -= 20
                    #imcv_edges.itemset(180,x,0,255)    # y,x, ch, data 
            
            global send_width
            global send_center
            global publisher
            global publisher2
            send_width = imcv_max-imcv_min
            send_center = (imcv_max+imcv_min)/2

            publisher.publish(str(send_width)) #type what you what to publish
            publisher2.publish(str(send_center)) 
            print("width:",send_width)
            print("center:",send_center)
            
            curr_time = time.time()
            exec_time = curr_time - prev_time
            info = "time:" + str(round(1000*exec_time, 2)) + " ms, FPS: " + str(round((1000/(1000*exec_time)),1))
            cv2.putText(imcv_edges, text=info, org=(50, 70),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255, 0, 0), thickness=2)
            cv2.imshow('Frame', imcv_edges)
            if cv2.waitKey(10) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    print('-> Done!')


if __name__ == '__main__':
    send_width = 0
    send_center = 0
    rospy.init_node('tutorial')

    publisher = rospy.Publisher('/DEPTHS', String, queue_size = 10)
    publisher2 = rospy.Publisher('/DEPTHS2', String, queue_size = 10)
    rate = rospy.Rate(20)

    args = parse_args()
    test_simple(args)

