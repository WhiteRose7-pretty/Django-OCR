from django.conf.urls import include, url
from django.urls import path, include

from . import views


urlpatterns = [
    path('', views.index, name='home'),
    path('home', views.home, name='home'),
    path('update', views.update_data, name='update'),
    path('detection', views.detection, name='detection'),
    path('get_data', views.getData, name='get_detection_data'),
    path('image_upload', views.imageUpload, name='image_upload'),
]