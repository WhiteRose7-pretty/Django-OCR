import mysql.connector
from django.conf import settings

if settings.SERVER == 'WINDOWS':
  mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="phpbb"
  )
else:
  mydb = mysql.connector.connect(
    host="localhost",
    user="fruitlog_db",
    password="dZYSPCaTW2c03qUg",
    database="fruitlog_db"
  )



