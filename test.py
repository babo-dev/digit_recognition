from keras.models import load_model
import numpy as np
import cv2

model = load_model("handwritten_digits.model")

image_number = 1
img = cv2.imread('image.png'.format(image_number))[:,:,0]
img = np.invert(np.array([img]))

prediction = model.predict(img)
final_pred = str(np.argmax(prediction))

print(final_pred)