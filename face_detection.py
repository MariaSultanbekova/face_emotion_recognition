from mtcnn.mtcnn import MTCNN
import cv2
import numpy
import imutils
import numpy as np


detector = MTCNN()


def get_coordinates(imageImage):
    image = np.array(imageImage)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    if image.shape[0] < image.shape[1]:  # приводим стороны к размеру 1000 пикселей
        image = imutils.resize(image, height=1000)
    else:
        image = imutils.resize(image, width=1000)
    image_size = numpy.asarray(image.shape)[0:2]


    faces_boxes = detector.detect_faces(image)  # координаты лиц


    coordinates = []
    if faces_boxes:

        face_n = 0
        for face_box in faces_boxes:
            face_n += 1

            x, y, w, h = face_box['box']
            # делаем отступы
            d = h - w
            w = w + d
            x = numpy.maximum(x - round(d / 2), 0)
            x1 = numpy.maximum(x - round(w / 4), 0)
            y1 = numpy.maximum(y - round(h / 4), 0)
            x2 = numpy.minimum(x + w + round(w / 4), image_size[1])
            y2 = numpy.minimum(y + h + round(h / 4), image_size[0])

            coordinates.append([int(x1), int(y1), int(x2), int(y2)])

    return coordinates
