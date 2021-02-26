import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt

test_box_path = "/Users/sherry/work/pycharm_python/reconstruction_ctpn/data/res/0.txt"
test_img_path = "/Users/sherry/work/pycharm_python/reconstruction_ctpn/data/test/0.png"


def get_box_by_file(box_path):
    def takeSecond(elem):
        return elem[0]

    f = open(box_path)  # 返回一个文件对象
    res = []
    line = f.readline()  # 调用文件的 readline()方法
    while line:
        # 112,110,512,110,512,131,112,131,0.99887985
        uv_s = line.split(",")
        xy = [int(_) for _ in uv_s[0:-1]]
        x = []
        y = []
        for i in range(len(xy)):
            if i % 2 == 0:
                x.append(xy[i])
            else:
                y.append(xy[i])
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        res.append([y_min, y_max, x_min, x_max])
        line = f.readline()
    f.close()
    res.sort(key=takeSecond, reverse=False)
    print(res)
    return res


def get_box_by_file_api(box_list):
    def takeSecond(elem):
        return elem[0]

    res = []
    for line in box_list:
        # 112,110,512,110,512,131,112,131,0.99887985
        xy = [int(_) for _ in line[0:-1]]
        x = []
        y = []
        for i in range(len(xy)):
            if i % 2 == 0:
                x.append(xy[i])
            else:
                y.append(xy[i])
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        res.append([y_min, y_max, x_min, x_max])
    res.sort(key=takeSecond, reverse=False)
    print(res)
    return res


def enlarge_box(box_list, amp=0.1):
    list = []
    for box in box_list:
        h = box[1]-box[0]
        temp0 = box[0] - int(h*amp)
        temp1 = box[1] + int(h*amp)
        temp2 = box[2] - int(h*amp)
        temp3 = box[3] + int(h*amp)
        list.append([temp0, temp1, temp2, temp3])
    return list


def cut_image_by_box(img, box_list):
    image_list = []
    # box_list = enlarge_box(box_list)
    for _ in box_list:
        cropped = img[_[0]:_[1], _[2]:_[3]]
        x, y = cropped.shape[0:2]
        r_x = 32. / x
        cropped = cv.resize(cropped, (0, 0), fx=r_x, fy=r_x)
        image_list.append(cropped)
        # cv.imshow(str(_), cropped)
        # cv.waitKey()
    return image_list


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h % 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w % 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def cut_and_sort_img(img_path, box_path):
    img = cv.imread(img_path)
    img, (rh, rw) = resize_image(img)
    box_list = get_box_by_file(box_path)
    image_list = cut_image_by_box(img, box_list)
    return image_list


def cut_and_sort_img_api(img, box_list):
    img, (rh, rw) = resize_image(img)
    box_list = get_box_by_file_api(box_list)
    image_list = cut_image_by_box(img, box_list)
    return image_list


def cut_and_show_crop(img_path, box_path):
    image_list = cut_and_sort_img(img_path, box_path)
    for img in image_list:
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        print(img_gray.shape)
        cv.imshow("cropped", img)
        cv.waitKey()


if __name__ == '__main__':
    cut_and_show_crop(test_img_path, test_box_path)
