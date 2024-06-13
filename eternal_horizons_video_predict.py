import cv2
import numpy as np
import tensorflow as tf
import imageio.v3 as iio
from pathlib import Path



class_names = ['animal_crossing', 'doom']

def init_model():
    return tf.keras.models.load_model('doomcrossing_tfv1.keras')

def read_image_dir():
    images = []
    file_names = []
    file_order = []

    for file in Path("./eternal-horizons-frames").iterdir():
        if not file.is_file():
            continue
        
        file_order.append(int(file.stem[5:]))
        file_names.append(file)

    file_names = np.array(file_names)
    file_numpy = np.array(file_order, dtype=int)

    indexes = file_numpy.argsort()
    file_names = file_names[indexes]

    for i in range(10):
        print(f"index: {indexes[i]} , f: {file_names[i]}")


    for file in file_names:
        if not file.is_file():
            continue
        img = iio.imread(file)
        img = np.expand_dims(img, axis=0)
        images.append(img)

    return images

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
bottomLeftCornerOfText = (5,200)
fontScale              = 0.5
fontColor              = (20,20,225)
thickness              = 2
lineType               = 2

def predict_to_video(video_name):
    model = init_model()
    images = read_image_dir()

    predicted_images = [
        (frame.squeeze(), predict_frame(model, frame))
        for frame in images
    ]

    for (im, lb) in predicted_images:
        im = cv2.putText(im, lb, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType) 
    height, width, layers = predicted_images[0][0].shape
    video = cv2.VideoWriter(filename=video_name, fourcc=0, frameSize=(width, height), fps=30)

    for (image, _) in predicted_images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()

predict_to_video('output.mp4')

# ffmpeg -i "DOOM CROSSING - Eternal Horizons.webm" -q:a 0 -map a eternal_horizons_audio.mp3
