import requests
import base64
import cv2
from PIL import Image
from io import BytesIO


# url = 'http://127.0.0.1:5000/api/v1/add'
# # files = [('image', open("C:/Users/thewa/Desktop/untitled1/1.jpg", 'rb'))]
# files = [('image', open("C:/Users/thewa/Desktop/hidayet_t.jpg", 'rb')), ('image', open("C:/Users/thewa/Desktop/hidayet-turkoglu.jpg", 'rb'))]
# x = requests.post(url, files=files)
# print(x.text)

url = 'http://127.0.0.1:5000/api/v1/recognize'
files = [('image', open("C:/Users/thewa/Desktop/hidayet_t_2.jpg", 'rb'))]
x = requests.post(url, files=files)
print(x.text)


# get_report returns bytes for the image of the report so, it should be converted correctly

# url = 'http://127.0.0.1:5000/api/v1/get_report'
# data = {"name": "Abdullah", "report_type": "individual", "period": "year", "week": "1", "month": "1", "year": "2020"}
# x = requests.post(url, data=data)
# stream = BytesIO(x.content)
# image = Image.open(stream).convert("RGBA")
# stream.close()
# image.show()


