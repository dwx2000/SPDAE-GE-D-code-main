################################################################################
# 本文件用于定义必要的数据操作函数
################################################################################
# 导入模块
import os
import cv2
import json
import numpy as np
from PIL import Image
from PIL import ImageOps
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
################################################################################
# 用于机械故障诊断数据集振动信号的截取
def data_intercept_for_mfd(data, sample_number, sample_size):
    data = data[:sample_size*sample_number, 0]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, 1))
    data = data.reshape((sample_number, sample_size))
    return data
################################################################################
# 用于收集图像数据建构图像集
def collect_images(path, file_list, new_width=20, new_height=20, is_HE=False):
    pixel_matrix = list()
    for file in file_list:
        file_path = os.path.join(path, file)
        image = Image.open(file_path)
        if is_HE:
            image = ImageOps.equalize(image)
        image = image.resize((new_width, new_height))
        image = image.convert("L")
        pixel_vector = np.array(image.getdata())
        pixel_matrix.append(pixel_vector)
    return np.array(pixel_matrix).T

################################################################################
# 用于数据存储的路径
def def_path(name, option="GRASSMANN", file_type=""):
    data_list = ["..", option, name, name + "_Data" + file_type]
    target_list = ["..", option, name, name + "_Target" + file_type]
    target_dict_list = ["..", option, name, name + "_Target_Dict.json"]
    save_list = ["..", option, name]
    save_path = "/".join(save_list)
    os.makedirs(save_path, exist_ok=True)
    return "/".join(data_list), "/".join(target_list), "/".join(target_dict_list)

def save_data(data, target, target_dict, data_path,
              target_path, target_dict_path):
    np.save(data_path, data)
    np.save(target_path, target)
    with open(target_dict_path, "w") as f:
        json.dump(target_dict, f, indent=4)
    f.close()

def save_grassmann_paras(data_name, grassmann_p, picture_height, picture_width):
    root = Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]
    leaf = ["DATA", "GRASSMANN", "Grassmann_data_paras.json"]
    root = list(root) + leaf
    paras_path = "/".join(root)
    with open(paras_path, 'r', encoding='utf-8') as paras:
        grassmann_paras = json.load(paras)
    paras.close()
    grassmann_paras["grassmann_p"][data_name] = grassmann_p
    grassmann_paras["picture_height"][data_name] = picture_height
    grassmann_paras["picture_width"][data_name] = picture_width
    grassmann_paras["high_dimensions"][data_name] = picture_height * picture_width
    with open(paras_path, 'w') as paras:
        json.dump(grassmann_paras, paras, indent=4)
    paras.close()
    save_data_name("GRASSMANN", data_name)

def save_data_name(options, data_name):
    root = Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]
    leaf = ["DATA", "data_options.json"]
    root = list(root) + leaf
    path = "/".join(root)
    with open(path, 'r', encoding='utf-8') as names:
        datas = json.load(names)
    names.close()
    if options not in datas.keys():
        datas[options] = []
    if data_name not in datas[options]:
        datas[options].append(data_name)
    with open(path, 'w') as names:
        json.dump(datas, names, indent=4)
    names.close()

def save_data_labels(data_name, abbre_labels):
    root = Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]
    leaf = ["DATA", "data_labels_abbre.json"]
    root = list(root) + leaf
    path = "/".join(root)
    with open(path, 'r', encoding='utf-8') as labels:
        labels_dict = json.load(labels)
    labels.close()
    labels_dict[data_name] = abbre_labels
    with open(path, 'w') as labels:
        json.dump(labels_dict, labels, indent=4)
    labels.close()


def convert_video(video_path, new_width=20, new_height=20):
    cap = cv2.VideoCapture(video_path)
    pixel_matrix = list()
    while cap.isOpened():
        flag, frame = cap.read()
        if flag:
            image = Image.fromarray(frame)
            image = image.resize((new_width, new_height))
            image = image.convert("L")
            pixel_vector = np.array(image.getdata())
            pixel_matrix.append(pixel_vector)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return np.array(pixel_matrix).T
def extract_frames(video_path, start_frame, end_frame, new_width=20, new_height=20):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pixel_matrix = list()
    for frame_number in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(frame)
        image = image.resize((new_width, new_height))
        image = image.convert("L")
        pixel_vector = np.array(image.getdata())
        pixel_matrix.append(pixel_vector)
    cap.release()
    return np.array(pixel_matrix).T
