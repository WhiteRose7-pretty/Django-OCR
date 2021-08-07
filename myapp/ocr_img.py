import cv2
import os
import json
import numpy as np
from django.conf import settings
import pytesseract
from pytesseract import Output

if settings.SERVER == 'WINDOWS':
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    tessdata_dir_config = r'--oem 1 --psm 10 --tessdata-dir "exam\\"'
else:
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    tessdata_dir_config = r'--oem 1 --psm 10 --tessdata-dir "exam/"'

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


def proc_ocr(img):
    info_text = pytesseract.image_to_string(img, lang='eng', config=tessdata_dir_config, output_type=Output.DICT)
    res_txt = info_text['text']
    # print('tess result : ' + res_txt)

    num = get_digit(res_txt)
    # print('extrated num : ' + num)
    return num

def ocr_colr(img_one):
    # img_bin = rgb_bin(img_one, 0)
    blue, green, red = cv2.split(img_one)
    num = proc_ocr(red)
    # show_img(red, False)
    if len(num) == 0:
        # img_bin = rgb_bin(img_one, 1)
        num = proc_ocr(green)
        # show_img(green, False)
        if len(num) == 0:
            # img_bin = rgb_bin(img_one, 2)
            num = proc_ocr(blue)
            # show_img(blue, False)

    if len(num) == 0:
        img_bin = cv2.threshold(red, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        num = proc_ocr(img_bin)
        # show_img(img_bin, False)
        if len(num) == 0:
            img_bin = cv2.threshold(green, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            num = proc_ocr(img_bin)
            if len(num) == 0:
                img_bin = cv2.threshold(blue, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                num = proc_ocr(img_bin)

    return num

def ocr_one(img_one):
    h, w = img_one.shape[:2]
    img_inv = 255 - img_one
    num = proc_ocr(img_inv)
    # print('org_num' + num)
    # show_img(img_inv, False)
    if len(num) > 0:
        return num

    img_bin = rgb_bin(img_one)
    num = proc_ocr(img_bin)
    # print('bin_num' + num)
    # show_img(img_bin, False)
    if len(num) > 0:
        return num

    img_re = cv2.resize(img_bin, (3*w, 3*h))
    num = proc_ocr(img_re)
    # print('resize_num' + num)
    # show_img(img_re, False)
    if len(num) > 0:
        return num

    img_re = cv2.resize(img_inv, (3*w, 3*h))
    num = proc_ocr(img_re)
    # print('inv_num' + num)
    # show_img(img_re, False)
    if len(num) > 0:
        return num

    if len(num) == 0:
        # img_bin = rgb_bin(img_one)
        # show_img(img_bin, False)
        img_re = cv2.resize(img_one, (3*w, 3*h))
        num = ocr_colr(img_re)
        if len(num) == 0:
            img_re = cv2.resize(img_inv, (6*w, 6*h))
            # num = proc_ocr(img_re)
            # show_img(img_re, False)
            num = ocr_colr(img_re)

            # img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            # img_bin = cv2.threshold(img_crop, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # img_bin = cv2.resize(img_bin, (w * 2, h * 2))

    # if len(num) == 0:
    #     # show_img(img_one, False)
    #     # img_re = cv2.resize(img_one, (2 * w, 2 * h))
    #     # show_img(img_re, False)
    #     # img_bin = rgb_bin(img_one)
    #     gray = cv2.cvtColor(img_one, cv2.COLOR_BGR2GRAY)
    #     img_bin = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #     show_img(img_one, False)
    #     num = proc_ocr(img_bin)
    #     print('rgb_bin : ' + num)
    #     show_img(img_bin, False)
    #     if len(num) == 0:
    #         img_re = cv2.resize(img_bin, (4 * w, 4 * h))
    #         num = proc_ocr(img_re)
    #         print('rgb_bin : ' + num)
    #         show_img(img_re, False)

    return num


def create_img_dot(img_in):
    # img_dot = create_img_dot(img_bin)
    # # img_inv = 255 - img_crop
    # show_img(img_crop, False)
    # if img_line is None:
    #     img_line = img_bin.copy()
    # else:
    #     img_line = img_concatenate(img_line, img_bin)
    # img_line = img_concatenate(img_line, img_dot)
    # num = proc_ocr(img_bin)
    # print(num)

    img = img_in.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, h), (255, 255, 255), -1)
    x = w // 9
    y = h // 9
    xx = 2 * x
    yy = 2 * y
    cv2.rectangle(img, (xx, yy), (7 * x, 3 * y), (0, 0, 0), -1)
    cv2.rectangle(img, (xx, 5 * y), (7 * x, 6 * y), (0, 0, 0), -1)

    # show_img(img, False)
    return img

def remove_overlap(img, rois):
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
            # cv2.rectangle(img, (xi, yi), (xi + wi, yi + hi), (255, 255, 255), 2)

    # show_img(img, False)
    # === }} remove the overlap rectangls
    return rect_roi

def proc_first_row(img, contours, roi):
    res = ''
    xx, yy, ww, hh = roi
    xfi, yfi, wfi, hfi = 0, 0, 0, 0
    roi_fi = []
    for i in range(0, len(contours)):
        c = contours[i]
        x, y, w, h = cv2.boundingRect(c)
        if y > yy - hh:
            continue
        if y < yy - 3 * hh:
            continue
        if x > xx - ww / 4 and x + w < xx + 2 * ww and w > ww:
            roi_fi.append((x, y, w, h))

    if len(roi_fi) == 0:
        return res

    xfi, yfi, wfi, hfi = roi_fi[0]
    for i in range(1, len(roi_fi)):
        x, y, w, h = roi_fi[i]
        if w > wfi:
            xfi, yfi, wfi, hfi = x, y, w, h

    img_first = img[yfi:yfi+hfi, xfi:xfi+wfi]
    num_fi = ocr_one(img_first)
    # show_img(img_first, False)

    h_im, w_im = img.shape[:2]
    img_line = img[yfi:yfi+hfi, xfi+wfi + 5:w_im-5]
    img_bin = rgb_bin(img_line)
    # show_img(img_line, False)
    # show_img(img_bin, False)

    kernel0 = np.ones((5, 15), np.uint8)
    img_dil = cv2.erode(img_bin, kernel0, iterations=1)
    # show_img(img_dil, False)

    rt_rois = []
    contours, hierarchy = cv2.findContours(img_dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in range(0, len(contours)):
        c = contours[i]
        x, y, w, h = cv2.boundingRect(c)
        if 3 * w > w_im:
            continue
        rt_rois.append((x,y,w,h))
        # cv2.rectangle(img_line, (x, y), (x + w, y + h), (255, 255, 255), 2)

    rt_letters = remove_overlap(img_line, rt_rois)
    # show_img(img_line, False)

    rt_letters = sorted(rt_letters, key=sort_x)
    len_let = len(rt_letters)
    if len_let == 0:
        return res

    for i in range(0, len_let):
        (x, y, w, h) = rt_letters[len_let - 1 - i]
        img_crop = img_line[y:y+h, x:x+w]
        num = ocr_one(img_crop)
        if len(num) > 0:
            res = res + num + ' '
        else:
            res = res + 'X' + ' '

    res = res + num_fi

    return res


kernel0 = np.ones((5, 5), np.uint8)
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
    # for (c, hier) in (contours, hierarchy):
    # for component in zip(contours, hierarchy):
    #     c = component[0]
    #     hier = component[1]
    for i in range(0, len(contours)):
        c = contours[i]
        hier = hierarchy[0][i]
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(gray_re, (x, y), (x + w, y + h), (255, 255, 255), 2)
        if w < 30 or h < 30:
            continue
        if w > 80 or h > 80:
            continue
        if hier[3] < 0:
            continue

        rois.append((x, y, w, h))

        # if w > 54 and w < 68 and h > 54 and h < 68:
        # if w > 45 and w < 75 and h > 35 and h < 75:
        #     rois.append((x, y, w, h))
        #     cv2.rectangle(gray_re, (x, y), (x + w, y + h), (192, 192, 192), 2)
        # if 2 * (y + h) < h_img:
        #     continue
        # if w > 30 and w < 110 and h > 10 and h < 30:
        #     rect_roi.append((x, y, w, h))
        #     cv2.rectangle(gray, (x, y), (x + w, y + h), (192, 192, 192), 2)
    # show_img(gray_re, False)

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
    rect_roi = remove_overlap(gray_re, rois)
    len_rts = len(rect_roi)
    if len_rts == 0:
        return res

    # === {{ get the rects in same line
    lines = []
    bover = [False] * len_rts
    for i in range(0, len_rts):
        if bover[i]:
            continue
        (xi, yi, wi, hi) = rect_roi[i]
        # cv2.rectangle(gray_re, (xi, yi), (xi + wi, yi + hi), (255, 255, 255), 2)
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
                # cv2.rectangle(gray_re, (xj, yj), (xj + wj, yj + hj), (255, 255, 255), 2)
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
    # show_img(gray_re)

    # print('line number : ' + str(len(lines)))
    lines = sorted(lines, key=sort_line)
    len_lines = len(lines)
    for i in range(0, len_lines):
        line = lines[len_lines - i - 1]
        len_rts = len(line)
        img_line = None
        for j in range(0, len_rts):
            (x, y, w, h) = line[len_rts - j - 1]
            img_crop = img_re[y:y+h, x:x+w]
            num = ocr_one(img_crop)
            # cv2.imwrite(dir_name + filename, img_crop)

            if len(num) > 0:
                res = res + num + ' '
            else:
                res = res + 'X' + ' '

            #     img_bin = cv2.resize(img_bin, (w * 4, h * 4))
            #     # im_bin = im_bin[5:h-5, 10:w-10]

        # if img_line is not None:
        #       show_img(img_line, False)
        #     num = proc_ocr(img_line)

    if len_lines > 0:
        line = lines[0]
        if len(line) > 0:
            res_first = proc_first_row(img_re, contours, line[0])
            res = res + res_first

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
