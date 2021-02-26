# coding=utf-8
from ctpn.main import infer_api as ctpn_infer_api
import cv2 as cv
from ocr_util import grabCut
from crnn import infer_api as crnn_infer_api

if __name__ == '__main__':
    im_ = cv.imread(r"/Users/sherry/Downloads/1.jpg")[:, :, ::-1]
    boxes = ctpn_infer_api.main(im_)
    image_list = grabCut.cut_and_sort_img_api(im_, boxes)
    image_list = [_ for _ in image_list if _.shape[1] > _.shape[0]]
    res_t = crnn_infer_api.main_api_list(image_list)
    for _ in res_t:
        print(_)
