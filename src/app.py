from flask import Flask, render_template, send_from_directory, url_for
import cv2
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from ultralytics import YOLO
import os 
import re
app = Flask(__name__)
app.config['SECRET_KEY'] = 'asldfkjlj'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos', IMAGES)

model = YOLO("YOLO_model/last.pt")

def load_img(path):
    print('PATH: ', path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def yolo_model(model):
    #model = YOLO("YOLO_model") #load yolo model train
    return model

def seg_img(model, img):
    detections = model.predict(img, project="static", name="predictions", save=True)
    #detections = detections[0].plot()

    return detections

configure_uploads(app,photos)
class UploadForm(FlaskForm):
    photo = FileField(

        validators=[
            FileAllowed(photos, 'Only images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')

@app.route('/static/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    detections = None
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        y_model = yolo_model(model)
        image = load_img(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        detections= seg_img(model=y_model,img=image)
        new_result_polygon = detections[0]
        extracted_masks = new_result_polygon.masks.data
        print('Dimension of the mask {}'.format(extracted_masks.shape))
        #Extract boxes, which likely contain class IDs
        detected_boxes = new_result_polygon.boxes.data
        #Extract class IDs from the detected boxes
        class_labels = detected_boxes[:,-1].int().tolist()
        # Initialize a dictionary to hold masks by class
        masks_by_class = {name: [] for name in new_result_polygon.names.values()}

        # Iterate through the masks and class labels
        for mask, class_id in zip(extracted_masks, class_labels):
            class_name = new_result_polygon.names[class_id]

            # Append the mask to the list in the dictionary
            masks_by_class[class_name].append(mask.cpu().numpy())

        # Initialize a list to store class names and the number of masks
        class_mask_counts = []

        # Iterate through the dictionary and print class names and counts
        for class_name, masks in masks_by_class.items():
            count = len(masks)
            class_mask_counts.append((class_name, count))
            print(f"Class Name: {class_name}, Number of Masks: {count}")

        # Print the list containing class names and counts
        print("Class Names and Number of Masks:", class_mask_counts)


            
        #print(detections)
        return render_template('index.html', form=form, file_url=file_url,
                           detections = detections, class_name = class_mask_counts)
    else:
        file_url = None    
    return render_template('index.html', form=form, file_url=file_url,
                           detections = detections, class_name = class_mask_counts)
if __name__ == "__main__":
    app.run(debug=True)