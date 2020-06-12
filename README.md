## FairFace

The paper: https://arxiv.org/abs/1908.04913

### Examples of FairFace Prediction
![](https://github.com/dchen236/FairFace/blob/master/examples/female.png width=100)
![](https://github.com/dchen236/FairFace/blob/master/examples/male.png width=100)

### Instructions to use FairFace

#### Download or Clone this repo

#### Install Dependencies
1. Please follow the [Pytorch's official documentation](https://pytorch.org/get-started/locally/) to install Pytorch
2. Please also install dlib, if you have pip installed on your system. Simply type the following command on your terminal.

```
pip install dlib
```

#### Download our models
Download our pretrained models from [here](https://drive.google.com/file/d/1SSfZLl-KoOkK_51cnk9S-Lm55g18mBX7/view?usp=sharing) and save it in the same folder as where predict.py is located. 

#### Prepare the images
prepare a csv and provide the paths of testing images where the colname name of testing images is "img_path" (see our [template csv file](https://github.com/dchen236/FairFace/blob/master/test_imgs.csv).


#### Run script predict.py
Run the predict.py script and provide the csv path (described in #Prepare-the-images section above).
```
python3 predict.py --csv "NAME_OF_CSV"
```
After download this repository, you can run `python3 predict.pu --csv test_imgs.csv`, the results will be available at (detected_
#### Results
The results will be saved at "test_outputs.csv" (located in the same folder as predict.py, see sample [here](https://github.com/dchen236/FairFace/blob/master/test_outputs.csv)

##### output file documentation
indices to type
- race_scores_fair (model confidence score)   [White, Black, Latino_Hispanic, East, Southeast Asian, Indian, Middle Eastern]
- race_scores_fair_4 (model confidence score) [White, Black, Latino_Hispanic, East, Southeast Asian, Indian, Middle Eastern]
- gender_scores_fair (model confidence score) [Male, Female]
- age_scores_fair (model confidence score)    [0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+]

### Data
Images (train + validation set): [Padding=0.25](https://drive.google.com/file/d/1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86/view), [Padding=1.25](https://drive.google.com/file/d/1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL/view)

We used dlib's get_face_chip() to crop and align faces with padding = 0.25 in the main experiments (less margin) and padding = 1.25 for the bias measument experiment for commercial APIs.
Labels: [Train](https://drive.google.com/file/d/1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH/view) [Validation](https://drive.google.com/file/d/1wOdja-ezstMEp81tX1a-EYkFebev4h7D/view)

License: CC BY 4.0

### Notes
The models and scripts were tested on a device with 8Gb GPU, it takes unders 2 seconds to predict the 5 images in test folder.
