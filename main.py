import os
# import matplotlib.pyplot as plt
import gdown
import insightface
from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
from faceswap import swap_n_show, swap_n_show_same_img, swap_face_single,fine_face_swap
import argparse
from queue import Queue
import cv2

parser = argparse.ArgumentParser(description="swap faces",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--mode', required=True, help='1 or 2 or 3')
parser.add_argument('-i', '--image_dir', default="image", help='image directory')
parser.add_argument('-o', '--out_dir', default="out", help='out directory')
args = parser.parse_args()

exts = ['.jpg', '.png', '.tif']


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)


def get_image_list_in_dir (dir):
    file_list = os.listdir(dir)
    file_list.sort()
    file_list_img = Queue()
    for file in file_list:
        fname, fext = os.path.splitext(file)
        if fext.lower() in exts:
            file_list_img.put(os.path.join(dir, file))
    return file_list_img


def init():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Download 'inswapper_128.onnx' file using gdown
    model_url = 'https://drive.google.com/uc?id=1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu'
    model_output_path = 'inswapper/inswapper_128.onnx'
    if not os.path.exists(model_output_path):
        gdown.download(model_url, model_output_path, quiet=False)

    swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)
    return app, swapper


def main(mode, image_dir, out_dir):
    app, swapper = init()

    create_folder(out_dir)

    image_list = get_image_list_in_dir(image_dir)

    if mode == '1':
        if image_list.qsize() < 2:
            print("Need at least 2 images in the image directory")
            return

        out_img = swap_face_single(image_list.get(), image_list.get(), app, swapper, enhance=False, enhancer='REAL-ESRGAN 2x',device="cpu")
    elif mode == '2' or mode == '3':
        if image_list.qsize() < 1:
            print("Need at least 1 images in the image directory")
            return
        out_img = swap_n_show_same_img(image_list.get(), app, swapper)


    # Save the image
    cv2.imwrite(os.path.join(out_dir, 'out.jpg'), out_img)


if __name__ == '__main__':
    main(args.mode, args.image_dir, args.out_dir)
