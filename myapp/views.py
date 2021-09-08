from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile
import base64
import time
from .ocr_img import detect_number
from django.conf import settings
# from .mysql import mydb
import mysql.connector


def detect_number_save(filepath):
    mydb = mysql.connector.connect(
        host="localhost",
        user="fruitlog_db",
        password="dZYSPCaTW2c03qUg",
        database="fruitlog_db"
    )
    res = detect_number(filepath)
    mycursor = mydb.cursor()
    sql = "INSERT INTO ocr_data (name, data) VALUES (%s, %s)"
    val = [
        (filepath, res),
    ]
    mycursor.executemany(sql, val)
    mydb.commit()
    mycursor.close()
    return res


def get_latest_data():
    mydb = mysql.connector.connect(
        host="localhost",
        user="fruitlog_db",
        password="dZYSPCaTW2c03qUg",
        database="fruitlog_db"
    )
    mycursor = mydb.cursor()
    sql = "SELECT * FROM ocr_data ORDER BY id DESC LIMIT 1;"
    mycursor.execute(sql)
    result = mycursor.fetchone()
    if result:
        data = result[2]
        id = result[0]
        temp = data.split(' : ')
    else:
        temp = ''
        id = ''
    mycursor.close()
    mydb.close()

    return temp, id


def index(request):
    return render(request, "index.html")


def update_data(request):
    data, id = get_latest_data()
    data = {
        'data':data,
        "id":id
    }
    return JsonResponse(data,  safe=False)


def detection(request):
    return render(request, "index.html", {})


def home(request):
    return render(request, "home.html", {})


def getData(request):
    st_time = int(round(time.time() * 1000))
    filepath = request.POST.get('path')
    filepath = filepath[1:]
    result = detect_number_save(filepath)
    print(int(round(time.time() * 1000)) - st_time)
    return JsonResponse({'status': 1, 'data': result})


def imageUpload(request):
    millis = int(round(time.time() * 1000))
    fs = FileSystemStorage()
    if request.method == 'POST' and request.POST.get('base_image'):
        image_data = request.POST.get('base_image')
        print("this is the base64 -------------------")
        format, imgstr = image_data.split(';base64,')
        ext = format.split('/')[-1]
        print(ext)
        if ext == 'JPG' or ext == 'jpg' or ext == 'png' or ext == 'PNG' or ext == 'jpeg' or ext == 'JPEG':
            data = ContentFile(base64.b64decode(imgstr))
            file_name = str(millis) + '.' + ext
            filename = fs.save(file_name, data)
            uploaded_file_url = fs.url(filename)
            print(uploaded_file_url)
            filepath = 'static' + uploaded_file_url
            result = detect_number_save(filepath)
            return JsonResponse({'status': 1, 'data': result})
        else:
            return JsonResponse({'status': 0, 'data': 'Invalid image. allowed jpg or png'})

    if request.method == 'POST' and request.FILES['file']:
        myfile = request.FILES['file']
        ext = myfile.name.split('.')[-1]
        print(ext)
        if ext == 'JPG' or ext == 'jpg' or ext == 'png' or ext == 'PNG' or ext == 'jpeg' or ext == 'JPEG':
            filename = fs.save(str(millis) + '.' + ext, myfile)
            uploaded_file_url = fs.url(filename)
            filepath = 'static' + uploaded_file_url
            result = detect_number_save(filepath)
            return JsonResponse({'status': 1, 'data': result})
        else:
            return JsonResponse({'status': 0, 'data': 'Invalid image. allowed jpg or png'})

    return JsonResponse({'status': 0, 'data': 'invalid'})


