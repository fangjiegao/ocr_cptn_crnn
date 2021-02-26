# coding=utf-8
from ctpn.main import infer_api as ctpn_infer_api
import cv2 as cv
from ocr_util import grabCut
from crnn import infer_api as crnn_infer_api

if __name__ == '__main__':
    res = []
    # [362, 486, 128, 656]
    im_ = cv.imread(r"/Users/sherry/work/pycharm_python/reconstruction_ctpn/data/test/c.jpg")[:, :, ::-1]
    boxes = ctpn_infer_api.main(im_)
    image_list = grabCut.cut_and_sort_img_api(im_, boxes)
    print("len:", len(image_list))
    for img in image_list:
        print("crnn_infer_api.main_api......")
        res_t = crnn_infer_api.main_api(img)
        res.append(res_t)
    print(res)
