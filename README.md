# FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age

The paper: https://openaccess.thecvf.com/content/WACV2021/papers/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.pdf

Karkkainen, K., & Joo, J. (2021). FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1548-1558).

### If you use our dataset or model in your paper, please cite:

```
 @inproceedings{karkkainenfairface,
  title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation},
  author={Karkkainen, Kimmo and Joo, Jungseock},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2021},
  pages={1548--1558}
}
```

### Examples of FairFace Prediction
![](https://github.com/dchen236/FairFace/blob/master/examples/female.png)
![](https://github.com/dchen236/FairFace/blob/master/examples/male.png)

### Instructions to use FairFace

- Download or Clone this repo
- Install Dependencies
   1. Please follow the [Pytorch's official documentation](https://pytorch.org/get-started/locally/) to install Pytorch
   2. Please also install dlib, if you have pip installed on your system. Simply type the following command on your terminal.

   ```
   pip install dlib
   ```
- Download our models
   Download our pretrained models from [here](https://drive.google.com/drive/folders/1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu?usp=sharing) and save it in the same folder as where predict.py is located. Two models are included, race_4 model predicts race as White, Black, Asian and Indian and race_7 model predicts races as White, Black, Latino_Hispanic, East, Southeast Asian, Indian, Middle Eastern.
- Unzip the downloaded FairFace model as well as dlib face detection models in dlib_models.
- Prepare the images
   - prepare a csv and provide the paths of testing images where the colname name of testing images is "img_path" (see our [template csv file](https://github.com/dchen236/FairFace/blob/master/test_imgs.csv).


### Run script predict.py
Run the predict.py script and provide the csv path (described in the section above).
```
python3 predict.py --csv "NAME_OF_CSV"
```
After download this repository, you can run `python3 predict.py --csv test_imgs.csv`, the results will be available at detected_faces (in case dlib detect multiple faces in one image, we save them here) and test_outputs.csv.
#### Results
The results will be saved at "test_outputs.csv" (located in the same folder as predict.py, see sample [here](https://github.com/dchen236/FairFace/blob/master/test_outputs.csv)

### UPDATES: 

### Run script predict_bbox.py
 same commands as predict.py, the output csv will have additional column "bbox" which is the bounding box of detected face.
```
python3 predict_bbox.py --csv "NAME_OF_CSV"
```
 

##### output file documentation
indices to type
- race_scores_fair (model confidence score)   [White, Black, Latino_Hispanic, East, Southeast Asian, Indian, Middle Eastern]
- race_scores_fair_4 (model confidence score) [White, Black, Asian, Indian]
- gender_scores_fair (model confidence score) [Male, Female]
- age_scores_fair (model confidence score)    [0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+]


### Data
Images (train + validation set): [Padding=0.25](https://drive.google.com/file/d/1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86/view), [Padding=1.25](https://drive.google.com/file/d/1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL/view)

We used dlib's get_face_chip() to crop and align faces with padding = 0.25 in the main experiments (less margin) and padding = 1.25 for the bias measument experiment for commercial APIs.
Labels: [Train](https://drive.google.com/file/d/1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH/view), [Validation](https://drive.google.com/file/d/1wOdja-ezstMEp81tX1a-EYkFebev4h7D/view)

License: CC BY 4.0

### Notes
The models and scripts were tested on a device with 8Gb GPU, it takes under 2 seconds to predict the 5 images in the test folder.
