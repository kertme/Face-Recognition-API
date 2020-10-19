import base64
import math
import os
import pickle
import random
import warnings
from calendar import monthcalendar
from datetime import datetime, timedelta
from os import path
import cv2
import dlib
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from system import Facenet
from imutils.face_utils import FaceAligner
from keras.preprocessing import image
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

warnings.filterwarnings("ignore")
dlib.DLIB_USE_CUDA = True

config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def new_detect_face(img, face_detector, face_aligner):
    exact_image = False
    if type(img).__module__ == np.__name__:
        exact_image = True

    base64_img = False
    if len(img) > 11 and img[0:11] == "data:image/":
        base64_img = True

    if base64_img:
        img = loadBase64Img(img)

    elif not exact_image:  # image path passed as input

        if not os.path.isfile(img):
            raise ValueError("Confirm that ", img, " exists")

        img = cv2.imread(img)

    img = imutils.resize(img, width=800)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 0)

    faces = []
    for rect in rects:
        aligned_face = face_aligner.align(img, gray, rect)
        img_pixels = image.img_to_array(aligned_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        # normalize input in [0, 1]
        img_pixels /= 255
        faces.append(img_pixels)

    return faces


def get_pickle(db_path, model, face_detector, face_aligner):
    if not os.path.isdir(db_path):
        raise ValueError("Passed db_path does not exist!")
    else:
        file_name = "representations.pkl"
        file_name = file_name.replace("-", "_").lower()

        if path.exists(db_path + "/" + file_name):
            f = open(db_path + '/' + file_name, 'rb')
            representations = pickle.load(f)
            return representations

        else:
            employees = []

            for r, d, f in os.walk(db_path):  # r=root, d=directories, f = files
                # if len(f) > 1:
                #     if ('.jpg' in f[0]) or ".JPG" in f[0]:
                #         exact_path = r + "/" + f[0]
                #         employees.append(exact_path)
                #     if ('.jpg' in f[1]) or ".JPG" in f[1]:
                #         exact_path = r + "/" + f[1]
                #         employees.append(exact_path)

                for file in f:
                    if ('.jpg' in file) or ".JPG" in file:
                        exact_path = r + "/" + file
                        employees.append(exact_path)

            if len(employees) == 0:
                raise ValueError("There is no image in ", db_path, " folder!")

            # ------------------------
            # find representations for db images

            representations = []

            pbar = tqdm(range(0, len(employees)), desc='Finding representations')

            # for employee in employees:
            for index in pbar:
                employee = employees[index]

                faces = new_detect_face(employee, face_detector, face_aligner)
                if len(faces) == 0:
                    continue
                img = faces[0]

                representation = model.predict(img)[0, :]

                #instance = [employee.split("/")[-2].split("\\")[-1], representation]
                instance = {"name": employee.split("/")[-2].split("\\")[-1], "representation": representation.tolist()}

                representations.append(instance)

            f = open(db_path + '/' + file_name, "wb")
            pickle.dump(representations, f)
            f.close()

            print("Representations stored in ", db_path, "/", file_name,
                  " file. Please delete this file when you add new identities in your database.")
            return representations


def loadBase64Img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def distance(a, b):
    x1, y1, x2, y2 = a[0], a[1], b[0], b[1]

    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def find(img_path, model, face_detector, face_aligner, representations, distance_metric='euclidean_l2'):
    resp_obj = []
    img_list = new_detect_face(img_path, face_detector, face_aligner)
    if not len(img_list):
        return resp_obj
    # raise ValueError(
    # 	"Face could not be detected. Please confirm that the picture is a face photo
    df = pd.DataFrame(representations)
    df.rename(columns={'name': 'identity'}, inplace=True)
    #df = pd.DataFrame(representations, columns=["identity", "representation"])

    for img in img_list:

        df_base = df.copy()

        # resp_obj = []

        target_representation = model.predict(img)[0, :]

        distances = []
        for index, instance in df.iterrows():
            source_representation = instance["representation"]
            source_representation = np.fromiter(source_representation, float)
            if distance_metric == 'cosine':
                distance = findCosineDistance(source_representation, target_representation)
            elif distance_metric == 'euclidean':
                distance = findEuclideanDistance(source_representation, target_representation)
            elif distance_metric == 'euclidean_l2':
                distance = findEuclideanDistance(l2_normalize(source_representation),
                                                     l2_normalize(target_representation))
            else:
                raise ValueError("Invalid distance_metric passed - ", distance_metric)

            distances.append(distance)

        # VGGFace thresholds
        # threshold = 0.40
        # if distance_metric == 'euclidean':
        #     threshold = 0.55
        # elif distance_metric == 'euclidean_l2':
        #     threshold = 0.75

        # Facenet thresholds
        threshold = 0.40
        if distance_metric == 'euclidean':
            threshold = 10
        elif distance_metric == 'euclidean_l2':
            threshold = 0.8

        df["distance"] = distances
        df = df.drop(columns=["representation"])
        df = df[df.distance <= threshold]

        df = df.sort_values(by=["distance"], ascending=True).reset_index(drop=True)
        resp_obj.append(df)
        df = df_base.copy()  # restore df for the next iteration

    return resp_obj


def test_db(db_path, representations, model, face_detector, face_aligner):
    if not os.path.isdir(db_path):
        raise ValueError("Passed db_path does not exist!")
    else:
        test_employees = []

        for r, d, f in os.walk(db_path):  # r=root, d=directories, f = files
            if len(f) > 2:
                if ('.jpg' in f[2]) or ".JPG" in f[2]:
                    exact_path = r + "/" + f[2]
                    test_employees.append(exact_path)

        pbar = tqdm(range(0, len(test_employees)), desc='Testing dataset')
        true, false, unknown = 0, 0, 0

        for index in pbar:
            employee = test_employees[index]
            list_df = find(img_path=employee, model=model, face_detector=face_detector, face_aligner=face_aligner,
                           representations=representations,
                           distance_metric="euclidean_l2")

            if len(list_df) == 1:
                df = list_df[0]
                if len(df) > 0:
                    if df.shape[0] > 0:
                        matched = df.iloc[0].identity
                        if matched.split("/")[-2].split("\\")[-1] == employee.split("/")[-2].split("\\")[-1]:
                            true += 1
                        else:
                            false += 1
                else:
                    unknown += 1
            elif len(list_df) == 0:
                unknown += 1
            else:
                false += 1

        print(f'true:{true} false:{false} unknown:{unknown}')
        print(f'accuracy:{100 * true / (true + false + unknown)}%')
        print(f'accuracy without unknowns:{(100 * true / (true + false)):.2f}%')


def find_input_shape(model):
    input_shape = model.layers[0].input_shape

    if type(input_shape) == list:
        input_shape = input_shape[0][1:3]
    else:
        input_shape = input_shape[1:3]

    target_shape = (input_shape[0], input_shape[1])
    return target_shape


def datetime_range(start=None, end=None):
    span = end - start
    for i in range(span.days + 1):
        yield start + timedelta(days=i)


# year: used for determination of the time period
# ex: year=2 -> generates fake data of current year and last year
def generate_fake_data(db, db_path, year):
    if not os.path.isdir(db_path):
        raise ValueError("Passed db_path does not exist!")
    else:
        db.drop()
        employees = [i for i in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, i))]
        now = datetime.now()
        end = datetime(int(now.strftime("%Y")), 12, 31)
        start = datetime(int(now.strftime("%Y")) - year + 1, 1, 1)

        for date in datetime_range(start, end):
            r_day = int(date.strftime("%d").lstrip("0"))
            r_month = int(date.strftime("%m").lstrip("0"))
            r_year = int(date.strftime("%Y"))
            for employee in employees:
                check_in_hour = random.randint(8, 12)
                check_in_minute = random.randint(0, 59)
                check_out_hour = random.randint(17, 19)
                check_out_minute = random.randint(0, 59)

                check_in_time = str(check_in_hour) + ":" + str(check_in_minute)
                check_out_time = str(check_out_hour) + ":" + str(check_out_minute)
                value = {"name": employee,
                         "year": r_year,
                         "month": r_month,
                         "day": r_day,
                         "check_in_time": check_in_time,
                         "check_out_time": check_out_time}
                db.insert_one(value)


def quick_plot(keys, values):
    colors = ["dodgerblue" if (x < max(values)) else "r" for x in values]
    highest_day = colors.index("r")
    y_pos = np.arange(0, 2 * len(keys), step=2)
    fig = plt.figure(figsize=(10, 5))
    plt.bar(y_pos, values, color=colors)
    plt.xticks(y_pos, keys)
    plt.ylabel('Work time in hours', fontsize=14)
    plt.yticks(np.arange(0., 16.1, 1.))
    plt.figtext(.76, .84, "Average Hour: %.1f" % np.mean(values))
    return fig, highest_day


def get_work_time(row):
    t = row["check_in_time"].split(":")
    check_in_time = int(t[0]) + int(t[1]) / 60
    t = row["check_out_time"].split(":")
    check_out_time = int(t[0]) + int(t[1]) / 60
    return check_out_time - check_in_time


def create_report(db, report_path=None, name=None, report_type="individual", period="month", week=1, month=1, year=1):
    global fig
    if report_path is None:
        report_path = os.path.abspath(os.getcwd())

    month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                  'August', 'September', 'October', 'November', 'December']
    day_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    if week < 1 or month < 1 or month > 12 or year < 0:
        raise ValueError("Date is not valid")

    if report_type == "individual":
        if period == "week":
            weeks = monthcalendar(year, month)

            if week > len(weeks):
                raise ValueError("Week number is not valid. It should be between 1 - %d for this month" % len(weeks))

            days_of_week = weeks[week - 1]
            start_of_week = min([day for day in days_of_week if day != 0])
            end_of_week = max(days_of_week)

            query = {'name': name, 'year': year, 'month': month,
                     'day': {'$gte': start_of_week, '$lte': end_of_week}}

            doc = db.find(query).sort("day")

            works = [get_work_time(day) for day in doc]

            values = works + [0] * (7 - len(works)) if days_of_week[0] != 0 \
                else list(filter(lambda x: x == 0, days_of_week)) + works

            fig, highest_day = quick_plot(day_list, values)
            plt.xlabel("Days", fontsize=14)
            plt.title("%d. week of %s %d report for %s" % (week, month_list[month - 1], year, name), fontsize=17)
            plt.text(2 * highest_day, values[highest_day] + 0.25, "Most worked day", color="r", ha='center')
            #plt.show()
            # pp = PdfPages(report_path + name + "_week" + str(week) +
            #               "_" + month_list[month - 1] + "_" + str(year) + "_report.pdf")

        elif period == "month":
            doc = db.find({"name": name, "year": year, "month": month}, {"_id": 0, "name": 0})
            works = {}
            for row in doc:
                works[row["day"]] = get_work_time(row)

            keys = list(works.keys())
            values = list(works.values())
            fig, highest_day = quick_plot(keys, values)
            plt.xlabel("Days", fontsize=14)
            plt.title("%s Report for %s" % (month_list[month - 1], name), fontsize=17)
            plt.text(2 * highest_day, values[highest_day] + 0.25, "Most worked day", color="r", ha='center')
            #plt.show()
            #pp = PdfPages(report_path + name + "_" + month_list[month - 1] + "_" + str(year) + "_report.pdf")
            # pp.attach_note("test note", positionRect=[10,10,10,10])

        elif period == "year":
            keys = list(map(lambda x: x[:3], month_list))
            values = []

            # query for grouped check in, check out times of months for person
            agr = [{'$match': {"$and": [{'year': year}, {'name': name}]}},
                   {'$group': {'_id': "$month",
                               'check_in_times': {'$push': "$check_in_time"},
                               'check_out_times': {'$push': "$check_out_time"}}},
                   {'$sort': {"_id": 1}}]

            doc = db.aggregate(agr)
            for month_dict in doc:
                # calculation of average check in and check out time (time format at db: "hour:minute")
                avg_check_in = np.mean(list(map(lambda x: int(x.split(":")[0]) + int(x.split(":")[1]) / 60,
                                                month_dict["check_in_times"])))
                avg_check_out = np.mean(list(map(lambda x: int(x.split(":")[0]) + int(x.split(":")[1]) / 60,
                                                 month_dict["check_out_times"])))
                values.append(avg_check_out - avg_check_in)

            fig, highest_day = quick_plot(keys, values)
            plt.xlabel("Months", fontsize=14)
            plt.title("%s Report for %s" % (year, name), fontsize=17)
            plt.text(2 * highest_day, values[highest_day] + 0.25, "Most worked month", color="r", ha='center')
            #plt.show()
            #pp = PdfPages(report_path + name + "_" + str(year) + "_report.pdf")

    elif report_type == "all":
        agr = [{'$group': {'_id': "$name"}}]
        employees = [row["_id"] for row in (db.aggregate(agr))]

        if period == "week":
            weeks = monthcalendar(year, month)

            if week > len(weeks):
                raise ValueError("Week number is not valid. It should be between 1 - %d for this month" % len(weeks))

            days_of_week = weeks[week - 1]
            start_of_week = min([day for day in days_of_week if day != 0])
            end_of_week = max(days_of_week)

            agr = [{'$match': {"$and": [{'year': year}, {'name': name},
                                        {'day': {'$gte': start_of_week, '$lte': end_of_week}}]}},
                   {'$group': {'_id': "$day",
                               'check_in_times': {'$push': "$check_in_time"},
                               'check_out_times': {'$push': "$check_out_time"}}},
                   {'$sort': {"_id": 1}}]

            doc = db.aggregate(agr)

            works = []
            for day_dict in doc:
                # calculation of average check in and check out time (time format at db: "hour:minute")
                avg_check_in = np.mean(list(map(lambda x: int(x.split(":")[0]) + int(x.split(":")[1]) / 60,
                                                day_dict["check_in_times"])))
                avg_check_out = np.mean(list(map(lambda x: int(x.split(":")[0]) + int(x.split(":")[1]) / 60,
                                                 day_dict["check_out_times"])))
                works.append(avg_check_out - avg_check_in)

            values = works + [0] * (7 - len(works)) if days_of_week[0] != 0 \
                else list(filter(lambda x: x == 0, days_of_week)) + works

            fig, highest_day = quick_plot(day_list, values)
            plt.xlabel("Days", fontsize=14)
            plt.title("%d. week of %s %d report for Company" % (week, month_list[month - 1], year), fontsize=17)
            plt.text(2 * highest_day, values[highest_day] + 0.25, "Most worked day", color="r", ha='center')
            #plt.show()
            # pp = PdfPages(report_path + "Week" + str(week) + "_" + month_list[month - 1] +
            #               "_" + str(year) + "_company_report.pdf")

        elif period == "month":
            doc = db.find({"month": month, "year": year},
                          {"_id": 0, "day": 1, "check_in_time": 1, "check_out_time": 1}).sort("day", 1)

            works = {}
            for row in doc:
                work_time = get_work_time(row)
                if row["day"] not in works:
                    works[row["day"]] = work_time
                else:
                    works[row["day"]] += work_time

            keys = list(works.keys())
            total_values = list(works.values())
            values = [x / len(employees) for x in total_values]
            fig, highest_day = quick_plot(keys, values)
            plt.xlabel("Days", fontsize=14)
            plt.title("%s Report for Company" % (month_list[month - 1]), fontsize=17)
            plt.text(2 * highest_day, values[highest_day] + 0.25, "Most worked day", color="r", ha='center')
            #plt.show()
            #pp = PdfPages(report_path + month_list[month - 1] + "_" + str(year) + "_company_report.pdf")

        elif period == "year":
            keys = list(map(lambda x: x[:3], month_list))
            values = []

            # query for grouped check in, check out times of months for all
            agr = [{'$match': {'year': year}},
                   {'$group': {'_id': "$month",
                               'check_in_times': {'$push': "$check_in_time"},
                               'check_out_times': {'$push': "$check_out_time"}}},
                   {'$sort': {"_id": 1}}]

            doc = db.aggregate(agr)
            for month_dict in doc:
                avg_check_in = np.mean(list(map(lambda x: int(x.split(":")[0]) + int(x.split(":")[1]) / 60,
                                                month_dict["check_in_times"])))
                avg_check_out = np.mean(list(map(lambda x: int(x.split(":")[0]) + int(x.split(":")[1]) / 60,
                                                 month_dict["check_out_times"])))
                values.append(avg_check_out - avg_check_in)

            fig, highest_day = quick_plot(keys, values)
            plt.xlabel("Months", fontsize=14)
            plt.title("%s Report for Company" % year, fontsize=17)
            plt.text(2 * highest_day, values[highest_day] + 0.25, "Most worked month", color="r", ha='center')
            # plt.show()
            #pp = PdfPages(report_path + str(year) + "_company_report.pdf")

    # pp.savefig(fig)
    # pp.close()
    return fig


def check_existence(db, name, date):
    doc = db.find({"name": name, "year": date.year, "month": date.month, "day": date.day},
                  {"_id": 0, "check_in_time": 1, "check_out_time": 1})
    check = False
    for row in doc:
        t = row["check_in_time"].split(":")
        check_in_date = datetime(date.year, date.month, date.day, int(t[0]), int(t[1]))
        t = row["check_out_time"].split(":")
        check_out_date = datetime(date.year, date.month, date.day, int(t[0]), int(t[1]))
        if check_in_date < date < check_out_date:
            check = True
            break
    return check


def live_demo(model, face_detector, face_aligner, representations):
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height
    print("Starting to analyse the live stream")
    frame = 1
    while True:
        ret, img = cam.read()
        list_df = find(img, model, face_detector, face_aligner, representations, distance_metric="euclidean_l2")

        detected_list = []
        for df in list_df:
            if len(df) > 0:
                if df.shape[0] > 0:
                    matched = df.iloc[0]
                    matched_name = matched.identity.split("/")[-2].split("\\")[-1]
                    matched_distance = matched.distance
                    # print(f'name:{matched_name} distance:{matched_distance}')
                    detected_list.append(matched_name)

        if len(detected_list) > 0:
            print(f'Frame:{frame} Recognized Faces: {detected_list}')
        frame += 1

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break


class System:
    def __init__(self):
        self.model = Facenet.loadModel()
        shape = find_input_shape(self.model)
        self.face_detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.getcwd() + "/system/shape_predictor_68_face_landmarks.dat")
        self.face_aligner = FaceAligner(predictor, desiredFaceWidth=shape[0], desiredFaceHeight=shape[1])

# myclient = pymongo.MongoClient("mongodb://localhost:27017/")
# mydb = myclient["mydatabase"]
# db = mydb["employees"]
# db_path = "C:/Users/thewa/Desktop/test-db/lfw_funneled"



# representations = get_pickle(db_path=db_path, model=model, face_detector=face_detector, face_aligner=face_aligner)
# test_db(db_path, representations, model, face_detector, face_aligner)
# live_demo(model, face_detector, face_aligner, representations)

# create_report(db, name="Abdullah", report_type="individual", period="week", week=2, month=1, year=2020)
# create_report(db, name="Abdullah", report_type="individual", period="month", week=2, month=1, year=2020)
# create_report(db, name="Abdullah", report_type="individual", period="year", week=2, month=1, year=2020)
# create_report(db, name="Abdullah", report_type="all", period="week", week=2, month=1, year=2020)
# create_report(db, name="Abdullah", report_type="all", period="month", week=2, month=1, year=2020)
# create_report(db, name="Abdullah", report_type="all", period="year", week=2, month=1, year=2020)
