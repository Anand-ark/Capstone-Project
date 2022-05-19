from django.shortcuts import render, HttpResponse
from django.core.files.storage import FileSystemStorage
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import joblib
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
cmodel=load_model('./models/covid_Xray_lungs_detection.h5')

img_height1,img_width1=150,150 #covid
def index(request):
    return render(request,'App/index.html',{})

def about(request):
    #return HttpResponse('<h1>Welcome </h1>')
    return render(request, 'App/about.html')
def contact(request):
    return render(request, 'App/contact.html')
def covid(request):
    return render(request, 'App/covid.html')

def upload1(request):#Covid
    p1 = request.FILES['image'];
    fs1=FileSystemStorage()
    filePathname1=fs1.save(p1.name,p1);
    filePathname1=fs1.url(filePathname1)
    testimage='.'+filePathname1
    img=image.load_img(testimage,target_size=(img_height1,img_width1))
    x=image.img_to_array(img)
    x=np.array(img)
    x=x/255;
    x=x.reshape(1,img_height1,img_width1,3)
    ans=cmodel.predict(x)
    if(ans[0][0]>ans[0][1]):
       ans='COVID POSITIVE DETECTED'
    else:
        ans='COVID NEGATIVE DETECTED'
    context={'filepathname1':filePathname1,'pred1':ans}
    return render(request, 'App/covidout.html',context)




