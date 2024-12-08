import datetime
import os
import cv2
from flask import Flask, Response, redirect, render_template, request, url_for
from dotenv import load_dotenv
from fashion_models import runFashionModels

load_dotenv()
global capture
capture=0

app =Flask(__name__, template_folder = 'templates', static_folder='static',static_url_path='/')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_pic', methods=['GET', 'POST'])
def capture_pic():
    print("here")
    global camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
    images = update_image_folder()
    caption = runFashionModels(images[0])
    return render_template('step1.html', images= images, caption=caption, active='capture_pic')


#make shots directory to save pics
try:
    os.mkdir('./static/imgs/shots')
except OSError as error:
    pass

camera = cv2.VideoCapture(0) #0 is computer, 1 is connected phone

def update_image_folder():
     # Define the folder where images are stored
    image_folder = 'static/imgs/shots'
    # List all image filenames in the folder
    images = sorted(os.listdir(image_folder))

    return images

def gen_frames():  # generate frame by frame from camera
    global capture
    while True:
        success, frame = camera.read() 
        frame = resize_and_crop_width(frame, 300, 600)
        if success:
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['static/imgs/shots', "latest_capture.png"])
                cv2.imwrite(p, frame)
                update_image_folder()
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass
    
def resize_and_crop_width(image, target_width, target_height):
    # Get original dimensions
    (h, w) = image.shape[:2]

    # Resize the image to match the target height while keeping aspect ratio
    aspect_ratio = w / h
    resized_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (resized_width, target_height), interpolation=cv2.INTER_AREA)

    # Crop the resized image to the target width
    if resized_width > target_width:
        # Calculate the center crop
        start_x = (resized_width - target_width) // 2
        cropped_image = resized_image[:, start_x:start_x + target_width]
    else:
        # If the resized width is smaller than target width, pad instead
        padding = (target_width - resized_width) // 2
        cropped_image = cv2.copyMakeBorder(resized_image, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return cropped_image


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='imgs/shots/' + filename))
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0')

