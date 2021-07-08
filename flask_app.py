from flask import Flask, render_template, request, flash, send_from_directory, redirect, url_for
from datetime import date
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from PIL import Image, ImageDraw
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
import os
import zipfile
import time
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'images'
ZIP_FOLDER = 'predict_zips'
PREDICTIONS = 'runs/hub/exp/'
SEMANTIC_SEGS = 'semantic_segmentations'
JSON_DIR = 'runs/hub/exp/'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
ALLOWED_ZIP_EXTENSIONS = set(['zip'])

def allowed_images(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_compressed_formats(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_ZIP_EXTENSIONS

def unzip_multi_file(filename):
    with zipfile.ZipFile(filename,"r") as zip_ref:
        zip_ref.extractall("targetdir")
        
def file_compress(inp_file_names, out_zip_file):
    
    compression = zipfile.ZIP_DEFLATED
    #print(f"Input File name passed for zipping - {inp_file_names}")
    print(f"out_zip_file is - {out_zip_file}")
    zf = zipfile.ZipFile(out_zip_file, mode="w")

    try:
        for file_to_write in inp_file_names:
            print(f"Processing file {file_to_write}")
            _, filename = os.path.split(file_to_write)
            zf.write(file_to_write, arcname=filename, compress_type=compression)
    except FileNotFoundError as e:
        print(f"Exception occurred during zip process - {e}")
    finally:
        zf.close()

def getTimestamp():
    timeStr = str(time.time())
    return timeStr.split('.')[0]

def decode_full_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for i in range(nc):
        idx = image == i
        r[idx] = label_colors[i, 0]
        g[idx] = label_colors[i, 1]
        b[idx] = label_colors[i, 2]

    rgb = np.stack([r, g, b], axis=2)

    return rgb


def create_pixel_mask(jpath: str, img_array: np.array):
    json_object = json.load(open(jpath))

    num_objects = len(json_object['index'])
    xmin_idx = json_object['columns'].index('xmin')
    xmax_idx = json_object['columns'].index('xmax')
    ymin_idx = json_object['columns'].index('ymin')
    ymax_idx = json_object['columns'].index('ymax')
    print('Number of Objects: ', num_objects)

    #Loop over all BBs, add 1 to the pixel value for every point within a BB
    for obj in json_object['data']:
        BB = [int(obj[xmin_idx]), int(obj[xmax_idx])+1, int(obj[ymin_idx]), int(obj[ymax_idx])+1]
        img_array[BB[2]:BB[3], BB[0]:BB[1]] += 1

    #Set all values of 2+, indicating they are within an overlap region, to zero
    x = np.where(img_array > 1)
    img_array[x] = 0

    return img_array

def decode_BB_segmap(image: np.array, label_dict: dict, label: str):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    if label:
        x, y = image.nonzero()
        label_color = label_dict[label]
        r[x, y] = label_color[0]
        g[x, y] = label_color[1]
        b[x, y] = label_color[2]

        rgb = np.stack([r,g,b], axis=2)
        return rgb


def find_overlap_region(BB1: list, BB2: list):
    #BB1 & BB2 are ordered: xmin, xmax, ymin, ymax
    BB1_xmin, BB1_xmax, BB1_ymin, BB1_ymax = BB1
    BB2_xmin, BB2_xmax, BB2_ymin, BB2_ymax = BB2

    overlap_region = {}

    #BB2 is completely within BB1 along the x-axis, overlap begins at BB2_xmin and ends at BB2_xmax
    if BB1_xmin < BB2_xmin and BB2_xmax < BB1_xmax:
        overlap_region['xmin'] = BB2_xmin
        overlap_region['xmax'] = BB2_xmax

    #BB2 overlaps with BB1 but extends past it, overlap begins at BB2_xmin and ends at BB1_xmax
    if BB1_xmin < BB2_xmin and BB2_xmin < BB1_xmax and BB1_xmax < BB2_xmax:
        overlap_region['xmin'] = BB2_xmin
        overlap_region['xmax'] = BB1_xmax

    #BB1 is completely within BB2 along the x-axis, overlap begins at BB1_xmin and ends at BB1_xmax
    if BB2_xmin < BB1_xmin and BB1_xmax < BB2_xmax:
        overlap_region['xmin'] = BB1_xmin
        overlap_region['xmax'] = BB1_xmax

    #BB1 overlaps with BB2 but extends past it, overlap begins at BB1_xmin and ends at BB2_xmax
    if BB2_xmin < BB1_xmin and BB1_xmin < BB2_xmax and BB2_xmax < BB1_xmax:
        overlap_region['xmin'] = BB1_xmin
        overlap_region['xmax'] = BB2_xmax

    if 'xmin' not in overlap_region:
        return None

    if 'xmax' not in overlap_region:
        return None

    #BB2 is completely within BB1 along the y-axis, overlap begins at BB2_ymin and ends at BB2_ymax
    if BB1_ymin < BB2_ymin and BB2_ymax < BB1_ymax:
        overlap_region['ymin'] = BB2_ymin
        overlap_region['ymax'] = BB2_ymax

    #BB2 overlaps with BB1 but extends past it, overlap begins at BB2_ymin and ends at BB1_ymax
    if BB1_ymin < BB2_ymin and BB2_ymin < BB1_ymax and BB1_ymax < BB2_ymax:
        overlap_region['ymin'] = BB2_ymin
        overlap_region['ymax'] = BB1_ymax

    #BB1 is completely within BB2 along the y-axis, overlap begins at BB1_ymin and ends at BB1_ymax
    if BB2_ymin < BB1_ymin and BB1_ymax < BB2_ymax:
        overlap_region['ymin'] = BB1_ymin
        overlap_region['ymax'] = BB1_ymax

    #BB1 overlaps with BB2 but extends past it, overlap begins at BB1_ymin and ends at BB2_ymax
    if BB2_ymin < BB1_ymin and BB1_ymin < BB2_ymax and BB2_ymax < BB1_ymax:
        overlap_region['ymin'] = BB1_ymin
        overlap_region['ymax'] = BB2_ymax

    if 'ymin' not in overlap_region:
        return None

    if 'ymax' not in overlap_region:
        return None

    return overlap_region

def semantic_segmentation(bouding_box_image):
    label_color_map = {'background': (0,0,0), 'aeroplane': (128,0,0), 'bicycle': (0,128,0), 'bird': (128,128,0),
                   'boat': (0,0,128), 'bottle': (128, 0, 128), 'bus': (0,128,128), 'car':(128,128,128), 'cat':(64,0,0),
                   'chair':(192,0,0), 'cow':(64,128,0), 'dining table':(192,128,0), 'dog':(64,0,128),
                   'horse':(192,0,128), 'motorbike':(64, 128, 128), 'person':(192,128,128), 'potted plant':(0,64,0),
                   'sheep':(128,64,0), 'sofa':(0,192,0), 'train':(128,192,0), 'tv/monitor':(0,64,128)}
    
    img_dir = UPLOAD_FOLDER
    semseg_dir = SEMANTIC_SEGS
    json_dir = JSON_DIR
    zip_dir = ZIP_FOLDER

    model1 = models.segmentation.fcn_resnet101(pretrained=True).eval()

    today = date.today()
    v = 'v0'
    #Loop over some of the images
    for imname in os.listdir(img_dir):
        print(imname)
        print('###### image to segment ########')
        print(imname)
        transform = T.Compose([T.Resize((512,512)),
                       T.ToTensor(),
                       T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])      

        #Create the "whole image mask"
        imcore = imname.split('.')[0]
        print('imcore: ', imcore)
        img = Image.open(f'./images/{imname}').convert('RGB')
        print('####### img print #######')
        print(img)

        inp = transform(img).unsqueeze(0)
        out1 = model1(inp)['out']
        print('Out shape: ', out1.shape)
        om = torch.argmax(out1.squeeze(), dim=0).detach().cpu().numpy()
        print ('om.shape: ', om.shape)
        print ('np.unique(om): ', np.unique(om))
        rgb = decode_full_segmap(om)
        rgb_im = Image.fromarray(np.uint8(rgb)).convert('RGB')
        one_shot_rgb_im = rgb_im.resize(img.size)
        print('img size: ', img.size)
        print('rgb_im size: ', rgb_im.size)
        new_img = Image.blend(img, one_shot_rgb_im, 0.45)
        plt.savefig('./'+imcore+'_segmap_whole_%s_%s.jpg' % (today, v))

        #Access the json file for this image
        json_name = imcore + '.json'
        json_path = json_dir + json_name
        json_object = json.load(open(json_path))

        #Get the relevant indices from the JSON file
        num_objects = len(json_object['index'])
        name_idx = json_object['columns'].index('name')
        xmin_idx = json_object['columns'].index('xmin')
        xmax_idx = json_object['columns'].index('xmax')
        ymin_idx = json_object['columns'].index('ymin')
        ymax_idx = json_object['columns'].index('ymax')
        print('Number of Objects: ', num_objects)

        #Create dummy BB combo mask
        combo_mask = Image.new('RGB', img.size)

        #Create a mask indicating whether each pixel is in a single BB (1), overlap region (0), or perimeter (0)
        np_img = np.asarray(img)
        print('np_img shape: ', np_img.shape)
        img_zeros_array = np.zeros((np_img.shape[0], np_img.shape[1]))
        print('img_zeros_array shape: ', img_zeros_array.shape)
        single_BB_pixels = create_pixel_mask(json_path, img_zeros_array)
        print('single_BB_pixels shape: ', single_BB_pixels.shape)
        single_BB_pixels_3d = np.stack([single_BB_pixels, single_BB_pixels, single_BB_pixels], axis=2)
        print('single_BB_pixels_3d shape: ', single_BB_pixels_3d.shape)

        #Process the individual BBs, create the combo mask
        for i, obj in enumerate(json_object['data']):
            name = obj[name_idx]
            obj_xmin = int(obj[xmin_idx])
            obj_xmax = int(obj[xmax_idx]) + 1 
            obj_ymin = int(obj[ymin_idx])
            obj_ymax = int(obj[ymax_idx]) + 1
            img_crop = img.crop((obj_xmin, obj_ymin, obj_xmax, obj_ymax))
            print('Image crop size: ', img_crop.size)
            #img_crop.save(semseg_dir+'/'+imcore+'_crop_%i_%s_%s.png' % (i, today, v), 'png')
            inp = transform(img_crop).unsqueeze(0)
            out = model1(inp)['out']
            print('BB out shape: ', out.shape)
            om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
            print ('BB om.shape: ', om.shape)
            print ('BB np.unique(om): ', np.unique(om))
            rgb = decode_BB_segmap(om, label_color_map, name)
            rgb_im = Image.fromarray(np.uint8(rgb)).convert('RGB')
            rgb_im = rgb_im.resize(img_crop.size)
            new_img_crop = Image.blend(img_crop, rgb_im, 0.45)
            plt.imshow(new_img_crop)
            plt.savefig(semseg_dir+'/'+imcore+'_segmap_%i_%s_%s.jpg' % (i, today, v))
            #plt.show()
            combo_mask.paste(rgb_im, (obj_xmin, obj_ymin))

    combo_img = Image.blend(img, combo_mask, 0.45)
    plt.imshow(combo_img)
    plt.savefig(semseg_dir+'/'+imcore+'_BB_combo_%s_%s.jpg' % (today, v))

    print('Creating final mask')
    np_final_mask = np.add(np.multiply((1 - single_BB_pixels_3d), np.array(one_shot_rgb_im)), np.multiply(single_BB_pixels_3d, np.array(combo_mask)))
    final_mask = Image.fromarray(np.uint8(np_final_mask)).convert('RGB')
    final_img = Image.blend(img, final_mask, 0.45)
    plt.savefig(semseg_dir+'/'+imcore+'_combo_final_%s_%s.jpg' % (today, v))

    zip_file_name = f'semantic_predictions_{getTimestamp()}.zip'
    print(zip_file_name)
    zip_path =  os.path.join(zip_dir, zip_file_name)

    file_name_list = []
    for files in os.listdir(semseg_dir):
        print(files)
        file_name_list.append(files)
    print('#### is this my error')
    file_compress(file_name_list, zip_path)

 



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ZIP_FOLDER'] = ZIP_FOLDER
app.config['JSON_DIR'] = JSON_DIR

def predict_label(img_path):
    img_dir = UPLOAD_FOLDER
    zip_dir = ZIP_FOLDER
    json_dir = JSON_DIR

    img_files = os.listdir(img_dir)
    zip_file_name = f'yolo_predictions_{getTimestamp()}.zip'
    print(zip_file_name)
    zip_path =  os.path.join(zip_dir, zip_file_name)

    model_s = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    threshold = 0.5
    results = model_s( img_dir+'/'+img_path)
    print(results)
    pd_results = results.pandas().xywh[0]
    print('Small model found %i objects' % len(pd_results))
    if len(pd_results) == 0 or pd_results['confidence'][0] < threshold:
        print('Small model failed, trying medium model')
        model_m = torch.hub.load('ultralytics/yolov5', 'yolov5m')
        mresults = model_m(img_dir+'/'+img_path)
        mpd_results = mresults.pandas().xywh[0]
        print('Medium model found %i objects' % len(mpd_results))
        if len(mresults) == 0 or mpd_results['confidence'][0] < threshold:
            model_l = torch.hub.load('ultralytics/yolov5', 'yolov5l')
            print('Medium model failed, trying large model')
            lresults = model_l(img_dir+'/'+img_path)
            lpd_results = lresults.pandas().xywh[0]
            print('Large model found %i objects' % len(lpd_results))
            if len(lresults) == 0 or lpd_results['confidence'][0] < threshold:
                print('Large model failed, trying XL model')
                model_x = torch.hub.load('ultralytics/yolov5', 'yolov5x')
                xresults = model_x(img_dir+'/'+img_path)
                xpd_results = lresults.pandas().xywh[0]
                print('XLarge model found %i objects' % len(xpd_results))
                fresults = xresults
            else:
                fresults = lresults 
        else:
            fresults = mresults 
    else:
        fresults = results

    fresults.save()
    json_results = fresults.pandas().xyxy[0].to_json(orient='split')
    parsed = json.loads(json_results)
    file_to_serv = f'runs/hub/exp/{img_path}'
    print('#### file to serv ####')
    print(file_to_serv)
    jsonString = json.dumps(parsed)
    image_id = img_path.split('.')[0]
    print('##### print image_id ######')
    print(image_id)
    
    jsonFile = open(f"runs/hub/exp/{image_id}.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    file_name_list = []
    file_name_list.append(file_to_serv)
    file_name_list.append(f'runs/hub/exp/{image_id}.json')
    file_compress(file_name_list, zip_path)
    return zip_file_name


@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit_single", methods = ['POST'])
def get_output_single():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        print('file was found, proceeding')
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_images(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            zip_name = predict_label(filename)
            semantic_file = semantic_segmentation(file)
            print(zip_name)
            return redirect(url_for('get_image', zip_name=zip_name))
        else:
            return render_template("wrong_file.html")

@app.route("/submit_multi", methods = ['POST'])
def get_output_multi():
     if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        print('file was found, proceeding')
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_compressed_formats(file.filename):
            multi_files = []
            
            unzip_multi_file(file.filename)
            # filename = secure_filename(file.filename)
            # img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # file.save(img_path)
            # p = predict_label(filename)
            
            return render_template("index.html", prediction = p, img_path = filename)
        return render_template("wrong_file.html")

@app.route("/get-zip/<zip_name>")
def get_image(zip_name):
    try:
        # return send_from_directory(app.config["ZIP_FOLDER"], filename=zip_name, as_attachment=True)
        return send_from_directory(app.config["ZIP_FOLDER"], path=zip_name)
    except FileNotFoundError:
        abort(404)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)