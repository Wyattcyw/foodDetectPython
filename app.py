import io
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import base64

app = Flask(__name__)
CORS(app)
model = YOLO("best.pt")

DETECTION_URL = "/yolov5"
IMAGE_URL = "/image"
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"
file_path = "/static/test.jpg"

@app.route("/", methods=["GET"])
def get():
    return("hello world!")

@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return
    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model.predict(source=img, save=False, conf=0.25)
        
        # Visualize the results
        # Load the image
        image = Image.open(io.BytesIO(image_bytes))
        plt.imshow(np.array(image))
        inferenceResult = []
        # Draw bounding boxes and labels
        for result in results:
            boxes = result.boxes.xyxy  # Bounding boxes
            confs = result.boxes.conf  # Confidence scores
            classes = result.boxes.cls  # Class indices
            
        for box, conf, cls in zip(boxes, confs, classes):
            # Draw the bounding box
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
       
            # Draw the label
            plt.text(x1, y1, f'{model.names[int(cls)]} {conf:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')
            plt.axis('off')
            
            resultObject = { "class" : model.names[int(cls)], "confs" : conf }
            inferenceResult.append(resultObject)
            resultObject['confs'] = resultObject['confs'].item()
            print(resultObject)
        # plt.show()
        # new_image = plt.savefig('my_plot.png')
        # base64_image = pyplot_to_blob(image)
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()
        data = {
            "image": my_base64_jpgData,
            "detections": inferenceResult,
        }
        # response = requests.post("http://newServiceIDk.com/sdfasf", json=data)
        # if response.status_code == 200:
        #     return jsonify({"message": "Image processed successfully"})
        # else:
        #     return jsonify({"error": f"Error processing image: {response.text}"}), 500
        return data
        # return jsonify(inferenceResult)
    else:
        return jsonify({"error": "No image file uploaded"}), 400  # Explicitly return 400 for bad request

if __name__ == "__main__":
    model = YOLO("best.pt")
    app.run(host="0.0.0.0", port=5000)
