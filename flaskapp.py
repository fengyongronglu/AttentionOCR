from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, os.path
import json
import re
import sys
import tarfile
import copy
import sys
import base64

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import textwrap
import numpy as np
from six.moves import urllib
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


from text_recognition import TextRecognition
from text_detection import TextDetection

from util import *
from shapely.geometry import Polygon, MultiPoint
from shapely.geometry.polygon import orient
from skimage import draw

# !flask/bin/python
from flask import Flask, jsonify, flash, Response
from flask import make_response
from flask import request, render_template
from flask_bootstrap import Bootstrap
from flask import redirect, url_for
from flask import send_from_directory

# from werkzeug import secure_filename
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from subprocess import call
# from sightengine.client import SightengineClient
import time

print("import finished")

UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'image'
VIDEO_FOLDER = r'video'
FOND_PATH = 'STXINWEI.TTF'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4','avi'])
VIDEO_EXTENSIONS = set(['mp4', 'avi'])
app = Flask(__name__)
bootstrap = Bootstrap(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.urandom(24)


# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def base64_to_image(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.frombuffer(img_data, np.uint8) # 注意，尽管读的深度图是16位，但是这里依然可以用8位进行字符串解析，结果跟16位一致，而且16位解析的话有些图会出错
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    return img

def init_ocr_model():
    detection_pb = './checkpoint/ICDAR_0.7.pb' # './checkpoint/ICDAR_0.7.pb'
    # recognition_checkpoint='/data/zhangjinjin/icdar2019/LSVT/full/recognition/checkpoint_3x_single_gpu/OCR-443861'
    # recognition_pb = './checkpoint/text_recognition_5435.pb' # 
    recognition_pb = './checkpoint/text_recognition.pb'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    with tf.device('/gpu:2'):
        tf_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True),#, visible_device_list="9"),
                                   allow_soft_placement=True)

        detection_model = TextDetection(detection_pb, tf_config, max_size=1600)
        recognition_model = TextRecognition(recognition_pb, seq_len=27, config=tf_config)
    label_dict = np.load('./reverse_label_dict_with_rects.npy', allow_pickle=True)[()] # reverse_label_dict_with_rects.npy  reverse_label_dict
    return detection_model, recognition_model, label_dict 

print("Init models...")
ocr_detection_model, ocr_recognition_model, ocr_label_dict = init_ocr_model()
print("Finished!")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS



##OCR TEST
@app.route('/', methods=['GET', 'POST'])
def ocr_upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        filename = file.filename
        if file and allowed_file(filename):
            if is_video(filename):
                file.save(os.path.join(app.config['VIDEO_FOLDER'], filename))

                return redirect(url_for('predict_ocr_video',
                                        filename=filename))
            else:
                filename = secure_filename(filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # fix_orientation(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return redirect(url_for('predict_ocr_image',
                                        filename=filename))
    #
    # return '''
    # <!doctype html>
    # <title>Upload new File for OCR</title>
    # <h1>Upload new File for OCR</h1>
    # <form method=post enctype=multipart/form-data>
    #   <input type=file name=file>
    #   <input type=submit value=Upload>
    # </form>
    # '''
    return render_template("ocr.html")


@app.route('/infer_raw_result', methods=["POST"])
def infer_raw_result():
    src_image_np = base64_to_image(request.json["image_base64"])
    print("infering...")
    r = detection(src_image_np, ocr_detection_model, ocr_recognition_model, ocr_label_dict, return_json=True)
    print("Finished!")
    return r

@app.route('/image_ocr/<filename>')
def predict_ocr_image(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    save_path = os.path.join(app.config['IMAGE_FOLDER'], filename)

    print("infering...")
    image = detection(img_path, ocr_detection_model, ocr_recognition_model, ocr_label_dict)
    print("Finished!")
    cv2.imwrite(save_path, image)
    return send_from_directory(app.config['IMAGE_FOLDER'],
                               filename)


@app.route('/video_ocr/<filename>')
def predict_ocr_video(filename):
    def stream_data():
        cap = cv2.VideoCapture(os.path.join(app.config['VIDEO_FOLDER'], filename))

        while True:
            ret, frame = cap.read()
            print(type(frame))
            if not ret:
                print('ret is False')
                break
            viz_image = detection_video(frame, ocr_detection_model, ocr_recognition_model, ocr_label_dict)
            viz_image = cv2.imencode('.jpg', viz_image)[1].tobytes()
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + viz_image + b'\r\n')

    return Response(stream_data(), mimetype='multipart/x-mixed-replace; boundary=frame')#redirect(out_path)


@app.route("/classify", methods=["POST"])
def classify():
    predictions = detection(request.data)
    print(predictions)
    return jsonify(predictions=predictions)



@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


from functools import reduce
import operator
import math


def order_points(pts):
    def centeroidpython(pts):
        x, y = zip(*pts)
        l = len(x)
        return sum(x) / l, sum(y) / l

    centroid_x, centroid_y = centeroidpython(pts)
    pts_sorted = sorted(pts, key=lambda x: math.atan2((x[1] - centroid_y), (x[0] - centroid_x)))
    return pts_sorted


def draw_annotation(image, points, label, horizon=True, vis_color=(30,255,255)):#(30,255,255)
    points = np.asarray(points)
    points = np.reshape(points, [-1, 2])
    cv2.polylines(image, np.int32([points]), 1, (0, 255, 0), 2)

    image = Image.fromarray(image)
    width, height = image.size
    fond_size = int(max(height, width)*0.03)
    FONT = ImageFont.truetype(FOND_PATH, fond_size, encoding='utf-8')
    DRAW = ImageDraw.Draw(image)

    points = order_points(points)
    if horizon:
        DRAW.text((points[0][0], max(points[0][1] - fond_size, 0)), label, vis_color, font=FONT)
    else:
        lines = textwrap.wrap(label, width=1)
        y_text = points[0][1]
        for line in lines:
            width, height = FONT.getsize(line)
            DRAW.text((max(points[0][0] - fond_size, 0), y_text), line, vis_color, font=FONT)
            y_text += height
    image = np.array(image)
    return image

def draw_annotation_modify(image, points, label, position, horizon=True, vis_color=(30,255,255)):#(30,255,255)
    # print(type(points)) #list
    points = np.asarray(points)
    # print(type(points)) #numpy
    points = np.reshape(points, [-1, 2])
    cv2.polylines(image, np.int32([points]), 1, (0, 255, 0), 2)
    # print(type(image)) #numpy.ndarray
    # print(points)

    image = Image.fromarray(image)
    # print(type(image)) #PIL.Image.Image
    width, height = image.size
    fond_size = int(max(height, width)*0.03)
    FONT = ImageFont.truetype(FOND_PATH, fond_size, encoding='utf-8')
    DRAW = ImageDraw.Draw(image)

    original_points = points
    points = order_points(points)
    print(type(points[0]))
    print(points)
    if horizon:
        DRAW.text((points[0][0], max(points[0][1] - fond_size, 0)), label, vis_color, font=FONT) #u, v
    else:
        lines = textwrap.wrap(label, width=1)
        y_text = points[0][1]
        for line in lines:
            width, height = FONT.getsize(line)
            DRAW.text((max(points[0][0] - fond_size, 0), y_text), line, vis_color, font=FONT)
            y_text += height
    image = np.array(image)

    # sorted_original_points = sorted(original_points, key=lambda x:x[0])
    # left_points = sorted_original_points[:2]
    # center = tuple(np.int32([(left_points[0][0] + left_points[1][0]) * 0.5, (left_points[0][1] + left_points[1][1]) * 0.5]))
    mean_array = np.mean(points, axis=0)
    left_points = [point for point in points if point[0] < mean_array[0]]
    left_points = sorted(left_points, key=lambda x:x[1])
    max_distance = 0.0
    center = np.array([0, 0])# (0, 0)error, center is not correctly found
    for i in range(len(left_points) - 1):
        distance = np.linalg.norm(left_points[i] - left_points[i+1])
        if distance > max_distance:
            max_distance = distance
            center = np.int32(0.5*(left_points[i] + left_points[i+1]))
    cv2.circle(image, tuple(center), 1, (0, 0, 255), 2)
    # cv2.circle(image, tuple(np.int32([100, 200])), 1, (0, 0, 255), 2)

    right_points = [point for point in points if point[0] > mean_array[0]]
    right_points = sorted(right_points, key=lambda x:x[1])
    max_distance = 0.0
    second_center = np.array([0, 0])# (0, 0)error, center is not correctly found
    for i in range(len(right_points) - 1):
        distance = np.linalg.norm(right_points[i] - right_points[i+1])
        if distance > max_distance:
            max_distance = distance
            second_center = np.int32(0.5*(right_points[i] + right_points[i+1]))
    cv2.circle(image, tuple(second_center), 1, (255, 0, 0), 2)

    position.append(center[0])
    position.append(center[1])
    position.append(second_center[0])
    position.append(second_center[1])
    for point in points:
        position.append(point[0])
        position.append(point[1])
    return image

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def mask_with_points(points, h, w):
    vertex_row_coords = [point[1] for point in points]  # y
    vertex_col_coords = [point[0] for point in points]

    mask = poly2mask(vertex_row_coords, vertex_col_coords, (h, w))  # y, x
    mask = np.float32(mask)
    mask = np.expand_dims(mask, axis=-1)
    bbox = [np.amin(vertex_row_coords), np.amin(vertex_col_coords), np.amax(vertex_row_coords),
            np.amax(vertex_col_coords)]
    bbox = list(map(int, bbox))
    return mask, bbox


def detection(img_path, detection_model, recognition_model, label_dict, it_is_video=False, return_json=False):
    if it_is_video:
        bgr_image = img_path
    elif isinstance(img_path, str):
        bgr_image = cv2.imread(img_path)
    elif isinstance(img_path, np.ndarray):
        bgr_image = img_path
    print(bgr_image.shape)
    vis_image = copy.deepcopy(bgr_image)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    total_time = 0.0
    start_time = time.time()
    r_boxes, polygons, scores = detection_model.predict(bgr_image)
    total_time += time.time() - start_time
    record = ""
    # r_boxes = [e.tolist() for e in r_boxes]
    # polygons = [e.tolist() for e in polygons]
    # scores = scores.tolist()
    # print("inference results:", len(r_boxes), len(polygons), len(scores))
    # print(r_boxes)
    # print(polygons)
    # print(scores)
    
    # if return_json:
    #     return json.dumps({"rotated_bboxes": r_boxes, "polygons": polygons, "scores": scores})

    for r_box, polygon, score in zip(r_boxes, polygons, scores):
        mask, bbox = mask_with_points(polygon, vis_image.shape[0], vis_image.shape[1])
        masked_image = rgb_image * mask
        masked_image = np.uint8(masked_image)
        cropped_image = masked_image[max(0, bbox[0]):min(bbox[2], masked_image.shape[0]),
                        max(0, bbox[1]):min(bbox[3], masked_image.shape[1]), :]

        height, width = cropped_image.shape[:2]
        test_size = 299
        if height >= width:
            scale = test_size / height
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            print(resized_image.shape)
            left_bordersize = (test_size - resized_image.shape[1]) // 2
            right_bordersize = test_size - resized_image.shape[1] - left_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=left_bordersize,
                                              right=right_bordersize, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.
        else:
            scale = test_size / width
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            print(resized_image.shape)
            top_bordersize = (test_size - resized_image.shape[0]) // 2
            bottom_bordersize = test_size - resized_image.shape[0] - top_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=top_bordersize, bottom=bottom_bordersize, left=0,
                                              right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.

        image_padded = np.expand_dims(image_padded, 0)
        print(image_padded.shape)

        start_time = time.time()
        results, probs = recognition_model.predict(image_padded, label_dict, EOS='EOS')
        total_time += time.time() - start_time
        # print(''.join(results))
        if len(results) == 0:
            continue
        print(''.join(str(result) for result in results if isinstance(result, str)))
        record += (''.join(str(result) for result in results if isinstance(result, str)))
        record += "\n"
        print(probs)
        record += (' '.join(map(str, probs)))
        record += "\n"

        ccw_polygon = orient(Polygon(polygon.tolist()).simplify(5, preserve_topology=True), sign=1.0)
        pts = list(ccw_polygon.exterior.coords)[:-1]
        positions = []
        # vis_image = draw_annotation(vis_image, pts, ''.join(results))
        vis_image = draw_annotation_modify(vis_image, pts, ''.join(results), position=positions) #resort again here
        record += ' '.join(map(str, positions))
        # print(record)
        record += "\n"

        # if height >= width:
        #     vis_image = draw_annotation(vis_image, pts, ''.join(results), False)
        # else:
        #     vis_image = draw_annotation(vis_image, pts, ''.join(results))
    if return_json:
        # return json.dumps({"rotated_bboxes": r_boxes, "polygons": polygons, "scores": scores})
        print("========")
        print(record)
        print("========")
        return json.dumps({"info": record})
    return vis_image


def detection_video(bgr_image, detection_model, recognition_model, label_dict):
    print(bgr_image.shape)
    vis_image = copy.deepcopy(bgr_image)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    r_boxes, polygons, scores = detection_model.predict(bgr_image)

    for r_box, polygon, score in zip(r_boxes, polygons, scores):
        mask, bbox = mask_with_points(polygon, vis_image.shape[0], vis_image.shape[1])
        masked_image = rgb_image * mask
        masked_image = np.uint8(masked_image)
        cropped_image = masked_image[max(0, bbox[0]):min(bbox[2], masked_image.shape[0]), max(0, bbox[1]):min(bbox[3], masked_image.shape[1]), :]

        height, width = cropped_image.shape[:2]
        test_size = 299
        if height>=width:
            scale = test_size/height
            resized_image = cv2.resize(cropped_image, (0,0), fx=scale, fy=scale)
            print(resized_image.shape)
            left_bordersize = (test_size - resized_image.shape[1]) // 2
            right_bordersize = test_size - resized_image.shape[1] - left_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=left_bordersize, right=right_bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
            image_padded = np.float32(image_padded)/255.
        else:
            scale = test_size/width
            resized_image = cv2.resize(cropped_image, (0,0), fx=scale, fy=scale)
            print(resized_image.shape)
            top_bordersize = (test_size - resized_image.shape[0]) // 2
            bottom_bordersize = test_size - resized_image.shape[0] - top_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=top_bordersize, bottom=bottom_bordersize, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
            image_padded = np.float32(image_padded)/255.

        image_padded = np.expand_dims(image_padded, 0)
        print(image_padded.shape)

        results, probs = recognition_model.predict(image_padded, label_dict, EOS='EOS')
        #print(''.join(results))
        print(probs)

        ccw_polygon = orient(Polygon(polygon.tolist()).simplify(5, preserve_topology=True), sign=1.0)
        pts = list(ccw_polygon.exterior.coords)[:-1]

        if height >= width:
            vis_image = draw_annotation(vis_image, pts, ''.join(results), False)
        else:
            vis_image = draw_annotation(vis_image, pts, ''.join(results))

    return vis_image

if __name__ == '__main__':
    # os.environ["TF_ENABLE_CONTROL_FLOW_V2"] = "0"
    app.run(host='0.0.0.0', port=8113, debug=False, threaded=True)
    # app.run("0.0.0.0", port=args.port, threaded = True)


