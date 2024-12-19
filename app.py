from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.secret_key = 'supersecretkey' 

model = load_model('model\Model_2 (1).keras')

CLASS_LABELS = ['Cataract', 'Conjunctivities', 'Normal']
def preprocess_image(image_path):
    """Load and preprocess the image to match the model's input shape."""
    image = load_img(image_path, target_size=(224, 224))  # Resize to 224x224
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image and make predictions
        image = preprocess_image(filepath)
        prediction = model.predict(image)
        predicted_class = CLASS_LABELS[np.argmax(prediction)]

        return render_template('index.html', prediction=predicted_class, image_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs('static/uploads/', exist_ok=True)
    app.run(debug=True)
