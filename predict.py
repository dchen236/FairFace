from __future__ import print_function, division
import warnings

warnings.filterwarnings("ignore")
import os.path
import os
import argparse
import logging
import time
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import pandas as pd
import numpy as np
import dlib


def rect_to_bb(rect):
    """Convert a dlib 'rect' object to a OpenCV-style bounding box

    Parameters
    ----------
    rect : dlib.rectangle
        A dlib rectangle object

    Returns
    -------
    tuple
        A 4-tuple (x, y, w, h) where (x, y) is the top-left corner of the
        bounding box and (w, h) are the width and height of the bounding box
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def load_dlib_models():
    cnn_face_detector = dlib.cnn_face_detection_model_v1(
        "dlib_models/mmod_human_face_detector.dat"
    )
    sp = dlib.shape_predictor("dlib_models/shape_predictor_5_face_landmarks.dat")
    return cnn_face_detector, sp


def resize_image(img, default_max_size=800):
    old_height, old_width, _ = img.shape
    if old_width > old_height:
        new_width, new_height = default_max_size, int(
            default_max_size * old_height / old_width
        )
    else:
        new_width, new_height = (
            int(default_max_size * old_width / old_height),
            default_max_size,
        )
    img = dlib.resize_image(img, rows=new_height, cols=new_width)
    return img


def find_faces_in_image(img, cnn_face_detector, sp):
    dets = cnn_face_detector(img, 1)
    faces = dlib.full_object_detections()
    for detection in dets:
        rect = detection.rect
        faces.append(sp(img, rect))
    return faces


def save_face_chips(images, image_path, detected_images_output_dir):
    for idx, image in enumerate(images):
        img_name = image_path.split("/")[-1]
        path_sp = img_name.split(".")
        face_name = os.path.join(
            detected_images_output_dir,
            path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1],
        )
        dlib.save_image(image, face_name)


def detect_face(
    image_paths,
    detected_images_output_dir,
    default_max_size=800,
    size=300,
    padding=0.25,
):
    logger = logging.getLogger(__name__)
    cnn_face_detector, sp = load_dlib_models()

    for index, image_path in enumerate(image_paths):
        if index % 1000 == 0:
            logger.info("---%d/%d---" % (index, len(image_paths)))

        img = dlib.load_rgb_image(image_path)
        img = resize_image(img, default_max_size)
        faces = find_faces_in_image(img, cnn_face_detector, sp)

        if len(faces) == 0:
            logger.info("Sorry, there were no faces found in '{}'".format(image_path))
            continue

        images = dlib.get_face_chips(img, faces, size=size, padding=padding)
        save_face_chips(images, image_path, detected_images_output_dir)


"""
def detect_face(
    image_paths, detected_images_output_dir, default_max_size=800, size=300, padding=0.25
):
    logger = logging.getLogger(__name__)
    cnn_face_detector = dlib.cnn_face_detection_model_v1(
        "dlib_models/mmod_human_face_detector.dat"
    )
    sp = dlib.shape_predictor("dlib_models/shape_predictor_5_face_landmarks.dat")
    for index, image_path in enumerate(image_paths):
        if index % 1000 == 0:
            logger.info("---%d/%d---" % (index, len(image_paths)))

        img = dlib.load_rgb_image(image_path)

        old_height, old_width, _ = img.shape

        if old_width > old_height:
            new_width, new_height = default_max_size, int(
                default_max_size * old_height / old_width
            )
        else:
            new_width, new_height = (
                int(default_max_size * old_width / old_height),
                default_max_size,
            )
        img = dlib.resize_image(img, rows=new_height, cols=new_width)

        dets = cnn_face_detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            logger.info("Sorry, there were no faces found in '{}'".format(image_path))
            continue
        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            rect = detection.rect
            faces.append(sp(img, rect))
        images = dlib.get_face_chips(img, faces, size=size, padding=padding)
        for idx, image in enumerate(images):
            img_name = image_path.split("/")[-1]
            path_sp = img_name.split(".")
            face_name = os.path.join(
                detected_images_output_dir,
                path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1],
            )
            dlib.save_image(image, face_name)
"""


def predidct_age_gender_race(save_prediction_at, imgs_path="cropped_faces/"):
    logger = logging.getLogger(__name__)
    img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(
        torch.load("fair_face_models/res34_fair_align_multi_7_20190809.pt")
    )
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    model_fair_4.load_state_dict(
        torch.load("fair_face_models/res34_fair_align_multi_4_20190809.pt")
    )
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.eval()

    trans = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # img pth of face images
    face_names = []
    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []
    race_scores_fair_4 = []
    race_preds_fair_4 = []

    for index, img_name in enumerate(img_names):
        if index % 1000 == 0:
            logger.info("Predicting... {}/{}".format(index, len(img_names)))

        face_names.append(img_name)
        image = dlib.load_rgb_image(img_name)
        image = trans(image)
        image = image.view(
            1, 3, 224, 224
        )  # reshape image to match model dimensions (1 batch size)
        image = image.to(device)

        # fair
        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)

        # fair 4 class
        outputs = model_fair_4(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:4]
        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        race_pred = np.argmax(race_score)

        race_scores_fair_4.append(race_score)
        race_preds_fair_4.append(race_pred)

    result = pd.DataFrame(
        [
            face_names,
            race_preds_fair,
            race_preds_fair_4,
            gender_preds_fair,
            age_preds_fair,
            race_scores_fair,
            race_scores_fair_4,
            gender_scores_fair,
            age_scores_fair,
        ]
    ).T
    result.columns = [
        "face_name_align",
        "race_preds_fair",
        "race_preds_fair_4",
        "gender_preds_fair",
        "age_preds_fair",
        "race_scores_fair",
        "race_scores_fair_4",
        "gender_scores_fair",
        "age_scores_fair",
    ]
    result.loc[result["race_preds_fair"] == 0, "race"] = "White"
    result.loc[result["race_preds_fair"] == 1, "race"] = "Black"
    result.loc[result["race_preds_fair"] == 2, "race"] = "Latino_Hispanic"
    result.loc[result["race_preds_fair"] == 3, "race"] = "East Asian"
    result.loc[result["race_preds_fair"] == 4, "race"] = "Southeast Asian"
    result.loc[result["race_preds_fair"] == 5, "race"] = "Indian"
    result.loc[result["race_preds_fair"] == 6, "race"] = "Middle Eastern"

    # race fair 4

    result.loc[result["race_preds_fair_4"] == 0, "race4"] = "White"
    result.loc[result["race_preds_fair_4"] == 1, "race4"] = "Black"
    result.loc[result["race_preds_fair_4"] == 2, "race4"] = "Asian"
    result.loc[result["race_preds_fair_4"] == 3, "race4"] = "Indian"

    # gender
    result.loc[result["gender_preds_fair"] == 0, "gender"] = "Male"
    result.loc[result["gender_preds_fair"] == 1, "gender"] = "Female"

    # age
    result.loc[result["age_preds_fair"] == 0, "age"] = "0-2"
    result.loc[result["age_preds_fair"] == 1, "age"] = "3-9"
    result.loc[result["age_preds_fair"] == 2, "age"] = "10-19"
    result.loc[result["age_preds_fair"] == 3, "age"] = "20-29"
    result.loc[result["age_preds_fair"] == 4, "age"] = "30-39"
    result.loc[result["age_preds_fair"] == 5, "age"] = "40-49"
    result.loc[result["age_preds_fair"] == 6, "age"] = "50-59"
    result.loc[result["age_preds_fair"] == 7, "age"] = "60-69"
    result.loc[result["age_preds_fair"] == 8, "age"] = "70+"

    result[
        [
            "face_name_align",
            "race",
            "race4",
            "gender",
            "age",
            "race_scores_fair",
            "race_scores_fair_4",
            "gender_scores_fair",
            "age_scores_fair",
        ]
    ].to_csv(save_prediction_at, index=False)

    logger.info("saved results at {}".format(save_prediction_at))


def ensure_dir(directory):
    """Check if directory exists, if not create it

    Parameters
    ----------
    directory : str
        path to directory
    """
    logger = logging.getLogger(__name__)
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info("Created directory {}".format(directory))
    else:
        logger.info("Directory {} already exists".format(directory))


def setup_logger(name, log_file, console_level=logging.INFO, file_level=logging.INFO):
    """Setup a logger that logs to console and file

    Parameters
    ----------
    name : str
        name of the logger

    log_file : str
        path to log file

    console_level : logging level
        level of logging to console

    file_level : logging level
        level of logging to file

    Returns
    -------
    logger : logging.logger
        logger object
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def setup_args():
    """Setup arguments for the script

    Returns
    -------
    parser : argparse.ArgumentParser
        parser object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="path to the csv file containing the image paths",
    )
    parser.add_argument(
        "--detected_faces_output",
        type=str,
        required=True,
        help="path to the output directory where the detected faces will be saved",
    )
    parser.add_argument(
        "--prediction_output",
        type=str,
        required=True,
        help="path to the output csv where the predictions will be saved",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA"
    )

    return parser


def main():
    """Main function"""
    logfile_output_dir = "./logs"
    ensure_dir(logfile_output_dir)

    logger = setup_logger(
        "main", os.path.join(logfile_output_dir, f"{time.time()}.log")
    )

    parser = setup_args()
    args = parser.parse_args()

    # TODO: explain why we need to do this and how to get the model
    os.environ["TORCH_HOME"] = "./torch_models"

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    dlib.DLIB_USE_CUDA = use_cuda
    logger.info("using CUDA?: %s" % use_cuda)

    ensure_dir(args.detected_faces_output)
    # Create a csv with one column 'img_path', which contains the full
    # paths of all images to be analyzed.
    imgs = pd.read_csv(args.input_csv)["img_path"]
    detect_face(imgs, args.detected_faces_output)

    logger.info("detected faces are saved at %s" % args.detected_faces_output)
    predidct_age_gender_race(args.prediction_output, args.detected_faces_output)


if __name__ == "__main__":
    main()
