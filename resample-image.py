import cv2
from cv2 import dnn_superres

model = 'FSRCNN'
magnification = 2
filename = 'Photo 10-10-2011 22 47 05'

img = cv2.imread(f'data/{filename}.jpg')

sr = dnn_superres.DnnSuperResImpl_create()

print(type(sr))

modelPath = f'models/{model}_x{magnification}.pb'

sr.readModel(modelPath)

sr.setModel('fsrcnn', magnification)

print('Upsampling')

result = sr.upsample(img)

print('Saving')

cv2.imwrite(f'data/{model}_x{magnification}_{filename}.png', result)
