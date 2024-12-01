import datetime
import os
import cv2
from flask import Flask, Response, render_template, request
from dotenv import load_dotenv

load_dotenv()
global capture
capture=0

app =Flask(__name__, template_folder = 'templates', static_folder='static',static_url_path='/')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/milestone1')
def interaction_1():
    global camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
    #images = update_image_folder()
    return render_template('index.html', active='interaction_1')


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
                p = os.path.sep.join(['static/imgs/shots', "shot_{}.png".format(str(now).replace(":",'_'))])
                p = p.replace(" ","_")
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
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0')

