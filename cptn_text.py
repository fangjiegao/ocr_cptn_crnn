# coding=utf-8
from ctpn.main import infer_api
import cv2 as cv
from ocr_util import grabCut

if __name__ == '__main__':
    # im_ = cv.imread(r"/Users/sherry/work/pycharm_python/reconstruction_ctpn/data/test/ocr.jpg")[:, :, ::-1]
    im_ = cv.imread(r"/Users/sherry/Downloads/1.jpg")[:, :, ::-1]
    boxes = infer_api.main(im_)
    image_list = grabCut.cut_and_sort_img_api(im_, boxes)
    for img in image_list:
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        print(img_gray.shape)
        cv.imshow("cropped", img)
        cv.waitKey()
