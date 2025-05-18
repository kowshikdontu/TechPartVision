import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

global curr
curr = {"description": "upload a image and click predict", "prediction_text": "None", "file": None}
model = tf.saved_model.load('pc_parts.h5')


def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    prediction = model.signatures['serving_default'](tf.constant(img_array))

    print("Output keys:", prediction.keys())

    return prediction


@app.route('/')
def home():
    return render_template('base.html')


@app.route('/render', methods=['POST', 'GET'])
def render():
    global curr
    if request.method == 'POST':
        f = request.files['image']
        print("got file")
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        filepath = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(filepath)
        curr["file"] = secure_filename(f.filename)

        preds = predict_image(filepath, model)

        # Print the prediction for debugging gives arr
        print("Prediction:", preds)

        description = {
            "cables": "Cables are essential components used to connect various hardware devices in a computer setup, including power cables, data cables, and peripheral cables.",
            "case": "The case, or chassis, houses and protects the computer's internal components, providing structure and cooling for the system.",
            "cpu": "The CPU (Central Processing Unit) is the brain of the computer, responsible for processing instructions and managing the operations of other components.",
            "gpu": "The GPU (Graphics Processing Unit) renders images and videos for display, handling tasks related to graphics and visual output.",
            "hdd": "The HDD (Hard Disk Drive) is a storage device used to store and retrieve digital information using magnetic storage.",
            "headset": "A headset combines headphones and a microphone, allowing for audio playback and voice communication, often used in gaming and telecommunication.",
            "keyboard": "The keyboard is an input device used to type and interact with a computer, consisting of keys for letters, numbers, and functions.",
            "microphone": "A microphone captures audio input, allowing users to record sound or communicate via voice.",
            "monitor": "The monitor is a display screen that shows the visual output from the computer, allowing users to interact with the graphical user interface.",
            "motherboard": "The motherboard is the main circuit board that connects all components of a computer, including the CPU, memory, and peripheral devices.",
            "mouse": "The mouse is a pointing device that allows users to interact with the computer's graphical user interface by moving a cursor on the screen.",
            "ram": "RAM (Random Access Memory) is a type of computer memory that stores data temporarily, allowing for fast access and efficient multitasking.",
            "speakers": "Speakers output audio from the computer, allowing users to hear sound from applications, media, and games.",
            "webcam": "A webcam captures video input, allowing users to participate in video calls, record videos, and take photos."
        }

        index = ['cables', 'case', 'cpu', 'gpu', 'hdd', 'headset', 'keyboard', 'microphone', 'monitor', 'motherboard',
                 'mouse', 'ram', 'speakers', 'webcam']
        curr["prediction_text"] = "The predicted item is: " + str(index[np.argmax(preds)])
        print(curr["prediction_text"])
        curr["description"] = description[str(index[np.argmax(preds)])]

        return render_template('prediction_page.html', text=curr["prediction_text"],
                               image=url_for('static', filename='uploads/' + curr["file"]),
                               description=curr["description"])
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
