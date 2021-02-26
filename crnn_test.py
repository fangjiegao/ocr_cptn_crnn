# coding=utf-8
import cv2 as cv
from crnn import infer_api as crnn_infer_api

if __name__ == '__main__':
    res = []
    img = cv.imread(r'/Users/sherry/work/pycharm_python/tf_crnn/src/data/demo/00000008.jpg')
    res_t = crnn_infer_api.main_api(img)
    res.append(res_t)
    print(res)
