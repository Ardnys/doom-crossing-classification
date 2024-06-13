import cv2
import numpy as np
import tensorflow as tf
import imageio.v3 as iio
from pathlib import Path

class_names = ['animal_crossing', 'doom']

def init_model():
    return tf.keras.models.load_model('doomcrossing_tfv2.keras')

def read_image_dir():
    images = []
    scaled_images = []
    file_names = []
    file_order = []

    # obtain the frames of the video somehow
    for file in Path("./eternal-horizons-frames").iterdir():
        if not file.is_file():
            continue
        
        file_order.append(int(file.stem[5:]))
        file_names.append(file)

    file_names = np.array(file_names)
    file_numpy = np.array(file_order, dtype=int)

    indexes = file_numpy.argsort()
    file_names = file_names[indexes]

    for file in file_names:
        if not file.is_file():
            continue
        img = iio.imread(file)
        scaled_img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        scaled_img = np.expand_dims(scaled_img, axis=0)

        scaled_images.append(scaled_img)
        images.append(img)

    return (images, scaled_images)

def predict_frame(model, frame) -> str:
    if not model:
        print("model has not been loaded")
        return ''
    prediction = model.predict(frame)
    prediction = tf.nn.sigmoid(prediction)
    prediction = np.argmax(prediction, axis=1)
    pred_class = class_names[prediction.squeeze()]
    return pred_class


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (25,1040)
fontScale              = 2
fontColor              = (20,20,225)
thickness              = 4
lineType               = 2

def predict_to_video(video_name):
    model = init_model()
    images, scaled_images = read_image_dir()

    predicted_images = [
        (frame.squeeze(), predict_frame(model, frame))
        for frame in scaled_images
    ]

    for ((pim, lb), im)  in zip(predicted_images, images):
        im = cv2.putText(im, lb, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType) 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = 1080, 1920
    video = cv2.VideoWriter(filename=video_name, fourcc=fourcc, frameSize=(width, height), fps=30)

    for image in images:
        video.write(image)

    print("Render complete.")
    cv2.destroyAllWindows()
    video.release()

predict_to_video('output.mp4')
