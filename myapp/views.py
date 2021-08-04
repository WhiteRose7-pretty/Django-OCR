from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile
import base64
import cv2
import time
import numpy as np
from imutils.object_detection import non_max_suppression
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
tessdata_dir_config = r'--oem 1 --psm 6 --tessdata-dir "exam/"'
import json


def index(request):
    # filepath = 'static/images/001.png'
    # result = detect_text(filepath)
    return render(request, "index.html", {})


def detection(request):
    # filepath = 'static/images/001.png'
    # result = detect_text(filepath)
    return render(request, "index.html", {})


# Create your views here.
def home(request):
    return render(request, "home.html", {})


def getData(request):
    st_time = int(round(time.time() * 1000))
    filepath = request.POST.get('path')
    filepath = filepath[1:]
    result = detect_number(filepath)
    # result = ''
    # result = detect_text(filepath)
    # result = result.split(',')
    # print(result)
    # re = json.dump(result)
    # print(re)
    print(int(round(time.time() * 1000)) - st_time)
    return JsonResponse({'status': 1, 'data': result})


def sort_x(rect):
    return rect[0]


def sort_y(rect):
    return rect[1]


def sort_line(line):
    return line[0][1]


def get_digit(text):
    num = ''
    arr_s = text.splitlines()
    for s in arr_s:
        if s.isdigit():
            num = s
            break
    return num


# image dection function
def detect_number(filepath):
    image = cv2.imread(filepath)
    if image is None:
        return 'Invalid iamge'

    res = ''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]
    gray_re = cv2.resize(gray, (w_img * 3, h_img * 3))
    blurred = cv2.GaussianBlur(gray_re, (9, 9), 0)
    # Adaptive threshold using 11 nearest neighbour pixels
    # imgbin = cv2.threshold(gray, 0, 150, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    img_bin = cv2.adaptiveThreshold(blurred, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    # cv2.imshow('img_gray', img_bin)
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow('img_time', img_dilate)
    # cv2.waitKey(0)

    # get the candinate regions of text line.
    rect_roi = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 54 and w < 68 and h > 54 and h < 68:
            rect_roi.append((x, y, w, h))
            # cv2.rectangle(gray_re, (x, y), (x + w, y + h), (192, 192, 192), 2)
        # if 2 * (y + h) < h_img:
        #     continue
        # if w > 30 and w < 110 and h > 10 and h < 30:
        #     rect_roi.append((x, y, w, h))
        #     cv2.rectangle(gray, (x, y), (x + w, y + h), (192, 192, 192), 2)

    rect_roi = sorted(rect_roi, key=sort_x)

    lines = []
    len_rts = len(rect_roi)
    bover = [False] * len_rts
    for i in range(0, len_rts):
        if bover[i]:
            continue
        (xi, yi, wi, hi) = rect_roi[i]
        # cv2.rectangle(gray_re, (xi, yi), (xi + wi, yi + hi), (192, 192, 192), 2)
        line = []
        line.append((xi, yi, wi, hi))
        bover[i] = True
        for j in range(i + 1, len(rect_roi)):
            if bover[j]:
                continue
            (xj, yj, wj, hj) = rect_roi[j]
            if abs(yi - yj) < min(hi, hj):
                line.append((xj, yj, wj, hj))
                bover[j] = True
                # cv2.rectangle(gray_re, (xj, yj), (xj + wj, yj + hj), (192, 192, 192), 2)

        lines.append(line)
        # cv2.imshow('img_gray', gray_re)
        # cv2.waitKey(0)

    # print('line number : ' + str(len(lines)))
    lines = sorted(lines, key=sort_line)
    len_lines = len(lines)
    for i in range(0, len_lines):
        line = lines[len_lines - i - 1]
        len_rts = len(line)

        for j in range(0, len_rts):
            (x, y, w, h) = line[len_rts - j - 1]
            img_crop = gray_re[y:y + h, x:x + w]
            # cv2.imwrite(dir_name + filename, img_crop)
            img_inv = 255 - img_crop
            text = pytesseract.image_to_string(img_inv, lang='eng', config=tessdata_dir_config,
                                               output_type=Output.DICT)
            print("text", text)
            num = get_digit(text['text'])
            if len(num) > 0:
                res = res + num + ' '
            print("num", num)
    print(res)
    return res


# image upload
def imageUpload(request):
    millis = int(round(time.time() * 1000))
    fs = FileSystemStorage()
    if request.method == 'POST' and request.POST.get('base_image'):
        image_data = request.POST.get('base_image')
        print("this is the base64 -------------------")
        format, imgstr = image_data.split(';base64,')
        ext = format.split('/')[-1]
        print(ext)
        # result = detect_number(filepath)
        if (ext == 'JPG' or ext == 'jpg' or ext == 'png' or ext == 'PNG' or ext == 'jpeg' or ext == 'JPEG'):
            data = ContentFile(base64.b64decode(imgstr))
            file_name = str(millis) + '.' + ext
            filename = fs.save(file_name, data)
            uploaded_file_url = fs.url(filename)
            print(uploaded_file_url)
            filepath = 'static' + uploaded_file_url
            result = detect_text(filepath)
            return JsonResponse({'status': 1, 'data': result})
        else:
            return JsonResponse({'status': 0, 'data': 'Invalid image. allowed jpg or png'})

    if request.method == 'POST' and request.FILES['file']:
        myfile = request.FILES['file']
        ext = myfile.name.split('.')[-1]
        print(ext)
        if (ext == 'JPG' or ext == 'jpg' or ext == 'png' or ext == 'PNG' or ext == 'jpeg' or ext == 'JPEG'):
            filename = fs.save(str(millis) + '.' + ext, myfile)
            uploaded_file_url = fs.url(filename)
            filepath = 'static' + uploaded_file_url
            result = detect_number(filepath)
            # result = detect_text(filepath)
            return JsonResponse({'status': 1, 'data': result})
        else:
            return JsonResponse({'status': 0, 'data': 'Invalid image. allowed jpg or png'})
    return JsonResponse({'status': 0, 'data': 'invalid'})


# detect string from image (main engine)
def detect_text(imagepath):
    # load the input image and grab the image dimensions
    img = cv2.imread(imagepath)
    img_org = img.copy()
    h_img, w_img, c = img.shape
    gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

    # set the parameter for tesseract ocr
    # tessdata_dir_config = r'--oem 1 --psm 6 --tessdata-dir "E:\\_work_python\\exam_ocr\\"'
    tessdata_dir_config = r'--oem 1 --psm 6 --tessdata-dir "exam\\"'

    # get the recognized result
    # text = pytesseract.image_to_string(img, lang='eng+chi_sim+equ', config=tessdata_dir_config)
    d = pytesseract.image_to_data(img, lang='eng+chi_sim+equ', config=tessdata_dir_config, output_type=Output.DICT)

    num_line = 0
    txtResult = ""
    n_boxes = len(d['level'])

    word_x = []
    word_y = []
    word_w = []
    word_h = []
    word_txt = []
    array_result = []

    line_avg_h = 0
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        # if 3 * h > h_img:
        #	continue

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(d['word_num'][i]), (x, y), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if num_line == int(d['line_num'][i]):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, str(num_line), (x + int(w / 2), y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            num_line = num_line + 1
            line_avg_h = line_avg_h + h
            continue

        if int(d['conf'][i]) < 10:
            continue

        word_x.append(x)
        word_y.append(y)
        word_w.append(w)
        word_h.append(y)
        word_txt.append(d['text'][i])

        # get the output for file
        txt_word = "{0}".format(d['text'][i])
        txt_rect = "{0},{1},{2},{3}".format(x, y, x + w, y + h)
        txt_type = "text"
        txt = {'text': txt_word, 'type': txt_type, 'boundingBox': txt_rect}
        array_result.append(txt)
        txtResult = txtResult + str(i) + " : text = " + txt_word + txt_rect + ",\n"

    line_avg_h = int(line_avg_h / num_line)

    # ========== get the figures ===========
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)
    kernel = np.ones((5, 5), np.uint8)
    img_dilate = cv2.dilate(edges, kernel, iterations=1)
    # cv2.imshow('img edged', img_dilate)
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img_org, contours, -1, (0, 255, 0), 3)

    figure_x = []
    figure_y = []
    figure_w = []
    figure_h = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 2 * line_avg_h:
            continue
        figure_x.append(x)
        figure_y.append(y)
        figure_w.append(w)
        figure_h.append(h)
        crop_img = img_org[y:y + h, x:x + w]
        img_string = 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.png', crop_img)[1]).decode()
        txt_rect = "{0},{1},{2},{3}".format(x, y, x + w, y + h)
        txt_type = "figure"
        txt = {'img': img_string, 'type': txt_type, 'boundingBox': txt_rect}
        array_result.append(txt)
        cv2.rectangle(img_org, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return array_result

    # cv2.imshow('img contours', img_org)
    # cv2.waitKey(0)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imwrite("img_result.png", img)
