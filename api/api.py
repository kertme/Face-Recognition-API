import flask
from flask import request, jsonify, send_file, make_response
from flask_pymongo import PyMongo
from system import system as sys
from PIL import Image
import io
import numpy as np
import cv2
from io import BytesIO, StringIO
import base64
import matplotlib.pyplot
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config["MONGO_URI"] = "mongodb://localhost:27017/mydatabase"
app.config['MONGO_DBNAME'] = 'employees'


mongo = PyMongo(app)
db = mongo.db
col = mongo.db["employees"]
col2 = mongo.db["registered_faces"]
system = sys.System()
# faces = sys.new_detect_face("C:/Users/thewa/Desktop/untitled1/img.jpg", system.face_detector, system.face_aligner)
# img = faces[0]
# representation = system.model.predict(img)[0, :]


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Face Recognition and Reporting System</h1>
<p>A prototype API for face recognition on given image and reporting mechanism on work hours</p>'''


@app.route('/api/v1/add', methods=['POST'])
def add_person():
    if 'image' not in request.files:
        return jsonify({"Failure": "You must send a photo to add with 'image' keyword"})

    photo_files = request.files.getlist('image')

    list_to_add = []
    for photo_file in photo_files:
        name = photo_file.filename.split(".")[0]
        npimg = np.fromfile(photo_file, np.uint8)
        photo = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        faces = sys.new_detect_face(photo, system.face_detector, system.face_aligner)

        if len(faces) == 0:
            # return jsonify({"Failure": "Face could not be detected on given photo"})
            continue

        img = faces[0]
        representation = system.model.predict(img)[0, :]
        list_to_add.append({'name': name, 'representation': representation.tolist()})
    col2.insert_many(list_to_add)
    return jsonify({"Success": f'{len(list_to_add)} photo added'}), 201


# @app.route('/api/v1/add_all', methods=['POST'])
# def create_representations():
#     args = request.args
#     # print(args)  # For debugging
#     if 'path'not in args:
#         return jsonify({"Failure": "Photo path should be decelerated"})
#     path = args['path']
#
#     representations = sys.get_pickle(path, system.model, system.face_detector, system.face_aligner)
#     col2.insert_many(representations)
#
#     #return jsonify({"representations": representations}), 201
#     return jsonify({'Success': str(len(representations)) + " photo added"}), 201


@app.route('/api/v1/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({"Failure": "You must send a image to recognize with 'image' keyword"})

    image_file = request.files['image']
    npimg = np.fromfile(image_file, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    representations = col2.find({}, {'_id': 0})
    list_df = sys.find(image, system.model, system.face_detector, system.face_aligner, representations)

    detected_list = []
    for df in list_df:
        if len(df) > 0:
            if df.shape[0] > 0:
                matched = df.iloc[0]
                matched_name = matched.identity
                matched_distance = matched.distance
                detected_list.append({'matched_name': matched_name, 'distance_score': matched_distance})

    if len(detected_list) > 0:
        return jsonify(detected_list)
    else:
        return jsonify({'Failure': 'There is no match in registered faces'})


@app.route('/api/v1/get_report', methods=['GET', 'POST'])
def create_report():
    if request.method == 'GET':
        args = request.args
    else:
        args = request.form
    expected_args = ["name", "report_type", "period", "week", "month", "year"]

    if all(x in args for x in expected_args):
        name = args['name']
        report_type = args['report_type']
        period = args['period']
        week = int(args['week'])
        month = int(args['month'])
        year = int(args['year'])
        fig = sys.create_report(col, name=name, report_type=report_type, period=period, week=week, month=month, year=year)
        if fig:
            if request.method == 'GET':
                buf = BytesIO()
                fig.savefig(buf, format="png")
                data = base64.b64encode(buf.getbuffer()).decode("ascii")
                return f"<img src='data:image/png;base64,{data}'/>"
            else:
                output = BytesIO()
                fig.savefig(output)
                output.seek(0)
                return send_file(output, mimetype='image/png')

        else:
            return jsonify({'Failure': 'Something went wrong while reporting, check employee records exist'})
    else:
        return jsonify(
            {'Failure': 'The query must include (name, report_type, period, week, month, year) fields'})


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


app.run()
