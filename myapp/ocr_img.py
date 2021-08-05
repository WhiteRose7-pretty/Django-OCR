import cv2
import os
import json
import numpy as np
from django.conf import settings
import pytesseract
from pytesseract import Output

if settings.SERVER == 'WINDOWS':
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    tessdata_dir_config = r'--oem 1 --psm 6 --tessdata-dir "exam\\"'
else:
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    tessdata_dir_config = r'--oem 1 --psm 6 --tessdata-dir "exam/"'

font = cv2.FONT_HERSHEY_SIMPLEX

def show_img(img, bresize):
    if bresize:
        img_re = cv2.resize(img, (800, 600))
        cv2.imshow('show_image', img_re)
    else:
        cv2.imshow('show_image', img)
    cv2.waitKey(0)


def sort_x(rect):
    return rect[0]


def sort_y(rect):
    return rect[1]


def sort_line(line):
    return line[0][1]


# image dection function
def get_digit(text):
    num = ''
    arr_s = text.splitlines()
    for s in arr_s:
        if s == 'S' or s == 's':
            s = '5'
        if s.isdigit():
            num = s
            break
    return num


def img_concatenate(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img = None
    if h1 == h2:
        img = np.concatenate((img1, img2), axis=1)
    elif h1 > h2:
        img = img1[0:h2, 0:w1]
        img = np.concatenate((img, img2), axis=1)
    else:
        img = img2[0:h1, 0:w2]
        img = np.concatenate((img1, img), axis=1)

    return img


def rgb_bin(img):
    h_im, w_im = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(h_im):
        for j in range(w_im):
            b = img[i][j][0]
            g = img[i][j][1]
            r = img[i][j][2]

            gray[i][j] = 255
            # if b < 60 and g < 60 and r < 60:
            #     gray[i][j] = 255
            #     continue

            if b > 100 and g > 100 and r > 100:
                gray[i][j] = 0
                continue

            if r > 100 and g < 50 and b < 50:
                gray[i][j] = 0
                continue

            if g > 100 and r < 50 and b < 50:
                gray[i][j] = 0
                continue

            if r < 10 and g > 45 and b > 45:
                gray[i][j] = 0
                continue

            # if img[i][j][0] < 60 and img[i][j][1] < 60 and img[i][j][2] < 60:
            #     gray[i][j] = 255 - img[i][j][0]
            # else :
            #     gray[i][j] = max(0, img[i][j][2] - 30)

    return gray
    # img_bin = cv2.adaptiveThreshold(gray, 255,
    #                                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 11)
    # return img_bin


kernel0 = np.ones((5, 5), np.uint8)


def proc_ocr(img):
    info_text = pytesseract.image_to_string(img, lang='eng', config=tessdata_dir_config, output_type=Output.DICT)
    res_txt = info_text['text']
    # print(res_txt)

    num = get_digit(res_txt)
    return num


def create_img_dot(img_in):
    img = img_in.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, h), (255, 255, 255), -1)
    x = w // 9
    y = h // 9
    xx = 2 * x
    yy = 2 * y
    cv2.rectangle(img, (xx, yy), (7 * x, 3 * y), (0, 0, 0), -1)
    cv2.rectangle(img, (xx, 5 * y), (7 * x, 6 * y), (0, 0, 0), -1)

    show_img(img, False)
    return img


def detect_number(filepath):
    image = cv2.imread(filepath)
    if image is None:
        return 'Invalid iamge'

    res = ''
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_re = cv2.resize(image, (w_img * 3, h_img * 3))
    gray_re = cv2.resize(gray, (w_img * 3, h_img * 3))
    # show_img(gray_re, False)

    blurred = cv2.GaussianBlur(gray_re, (5, 5), 0)
    img_canny = cv2.Canny(blurred, 30, 90)
    # show_img(img_canny, False)
    img_bin = cv2.dilate(img_canny, kernel0, iterations=1)

    # img_bin = cv2.threshold(gray_re, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # Adaptive threshold using 11 nearest neighbour pixels
    # imgbin = cv2.threshold(gray, 0, 150, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # img_bin = cv2.adaptiveThreshold(gray_re, 255,
    #                                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)
    # show_img(img_bin, False)

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # get the candinate regions of text line.
    rois = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 30 or h < 30:
            continue
        if w > 80 or h > 80:
            continue

        rois.append((x, y, w, h))
        # cv2.rectangle(gray_re, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # if w > 54 and w < 68 and h > 54 and h < 68:
        # if w > 45 and w < 75 and h > 35 and h < 75:
        #     rois.append((x, y, w, h))
        #     cv2.rectangle(gray_re, (x, y), (x + w, y + h), (192, 192, 192), 2)
        # if 2 * (y + h) < h_img:
        #     continue
        # if w > 30 and w < 110 and h > 10 and h < 30:
        #     rect_roi.append((x, y, w, h))
        #     cv2.rectangle(gray, (x, y), (x + w, y + h), (192, 192, 192), 2)
    # cv2.imshow('img_gray', gray_re)
    # cv2.waitKey(0)

    # rois = sorted(rois, key=sort_y)
    # for i in range(0, len(rois)):
    #     (x, y, w, h) = rois[i]
    #     # img_crop = gray_re[y:y+h, x:x+w]
    #     img_crop = img_re[y:y + h, x:x + w]
    #     show_img(img_crop, False)
    #     img_bin = rgb_bin(img_crop)
    #     proc_ocr(img_bin)
    #     show_img(img_bin, False)

    rois = sorted(rois, key=sort_x)
    rect_roi = []
    # === {{ remove the overlap rectangls
    len_rts = len(rois)
    for i in range(0, len_rts):
        bOverlap = False
        (xi, yi, wi, hi) = rois[i]
        for j in range(0, len_rts):
            if i == j:
                continue
            (xj, yj, wj, hj) = rois[j]
            if xj <= xi and yj <= yi and xj + wj >= xi + wi and yj + hj >= yi + hi:
                bOverlap = True
                break
        if bOverlap == False:
            rect_roi.append((xi, yi, wi, hi))
            # cv2.rectangle(gray_re, (xi, yi), (xi + wi, yi + hi), (255, 255, 255), 2)
        # cv2.imshow('img_gray', gray_re)
        # cv2.waitKey(0)
    # === }} remove the overlap rectangls

    # === {{ get the rects in same line
    lines = []
    len_rts = len(rect_roi)
    bover = [False] * len_rts
    for i in range(0, len_rts):
        if bover[i]:
            continue
        (xi, yi, wi, hi) = rect_roi[i]
        cv2.rectangle(gray_re, (xi, yi), (xi + wi, yi + hi), (255, 255, 255), 2)
        line = []
        line.append((xi, yi, wi, hi))
        roi_check = []
        roi_check.append(i)
        for j in range(i + 1, len(rect_roi)):
            if bover[j]:
                continue
            (xj, yj, wj, hj) = rect_roi[j]
            if abs(xj - xi - wi) * 2 > max(wi, wj):
                continue
            if abs(yi - yj) < min(hi, hj) and abs(yi + hi - yj - hj) < min(hi, hj):
                line.append((xj, yj, wj, hj))
                (xi, yi, wi, hi) = (xj, yj, wj, hj)
                # bover[j] = True
                roi_check.append(j)
                cv2.rectangle(gray_re, (xj, yj), (xj + wj, yj + hj), (255, 255, 255), 2)
        if len(line) > 8:
            print('line number : ' + str(len(line)))
            # print('first rect : ' + str(line[0]))
            lines.append(line)
            for j in roi_check:
                bover[j] = True

    # show_img(gray_re, True)
    # return res
    # === }} get the rects in same line

    # gray_re = 255 - gray_re
    # cv2.imshow('img_crop', gray_re)
    # cv2.waitKey(0)

    # print('line number : ' + str(len(lines)))
    lines = sorted(lines, key=sort_line)
    len_lines = len(lines)
    for i in range(0, len_lines):
        line = lines[len_lines - i - 1]
        len_rts = len(line)
        img_line = None
        for j in range(0, len_rts):
            (x, y, w, h) = line[len_rts - j - 1]
            # img_crop = gray_re[y+2:y+h-4, x+2:x+w-4]
            img_crop = img_re[y:y + h, x:x + w]
            # show_img(img_crop, False)

            # cv2.imwrite(dir_name + filename, img_crop)

            img_bin = rgb_bin(img_crop)
            # img_bin = cv2.threshold(img_crop, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            img_bin = cv2.resize(img_bin, (w * 2, h * 2))
            # show_img(img_bin, False)

            # img_dot = create_img_dot(img_bin)
            # # img_inv = 255 - img_crop
            # # cv2.imshow('img_crop', img_crop)
            # # cv2.waitKey(0)
            # if img_line is None:
            #     img_line = img_bin.copy()
            # else:
            #     img_line = img_concatenate(img_line, img_bin)
            # img_line = img_concatenate(img_line, img_dot)

            num = proc_ocr(img_bin)
            # print(num)
            if len(num) > 0:
                res = res + num + ' '
                # print(num)
            else:
                # show_img(img_bin, False)
                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                img_bin = cv2.threshold(img_crop, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                img_bin = cv2.resize(img_bin, (w * 2, h * 2))
                # show_img(img_bin, False)
                num = proc_ocr(img_bin)
                print(num)
                if len(num) > 0:
                    res = res + num + ' '
                else:
                    res = res + 'X' + ' '
                    print('X')

            #     img_bin = cv2.resize(img_bin, (w * 4, h * 4))
            #     # im_bin = im_bin[5:h-5, 10:w-10]

            #     text = pytesseract.image_to_string(img_bin, lang='eng', config=tessdata_dir_config, output_type=Output.DICT)
            #     num = get_digit(text['text'])
            #     if len(num) > 0:
            #         res = res + num + ' '
            #         print(num)

        # if img_line is not None:
        #     cv2.imshow('img_line', img_line)
        #     cv2.waitKey(0)
        #     num = proc_ocr(img_line)

    print(res)

    return res


def processImage(dir_name, fname):
    txtValue = ''
    # print("=========================================================")
    # print("   File name : " + fname)
    # print("=========================================================")
    if fname.endswith(('.jpeg', '.jpg', '.png', '.gif', '.tif', '.JPEG')):
        fpath = dir_name + fname

        detect_number(fpath)

    return txtValue


if __name__ == '__main__':
    filenames = []
    dir_name = '0729/'
    if dir_name:
        directory = os.fsencode(dir_name)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            filenames.append(filename)

    res_json = {}
    res_string = ''
    if len(filenames) > 0:
        for fname in filenames:
            txtValue = processImage(dir_name, fname)
            fname.replace('.png', '')
            res_json[fname] = txtValue

        res_string = json.dumps(res_json)
    else:
        print("   The files are not existed in folder.")

    with open("Output.txt", "w", 5, "utf-8") as text_file:
        text_file.write(str(res_string))
