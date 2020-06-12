## FairFace



### Instructions of using the models to predict race, gender and age

#### Download or Clone this repo
#### Install Dependencies

1. Please follow the [Pytorch's official documentation](https://pytorch.org/get-started/locally/) to install Pytorch
2. Please also install dlib, if you have pip installed on your system. Simply type the following command on your terminal.

```
pip install dlib
```

#### Download our models

Download our pretrained models from [here](https://drive.google.com/file/d/1SSfZLl-KoOkK_51cnk9S-Lm55g18mBX7/view?usp=sharing) and save it in the same folder as predict.py is located. 

#### Prepare the images

prepare a csv and provide the paths of testing images where the colname name of testing images is "img_path" (see our [template csv file](https://github.com/dchen236/FairFace/blob/master/test_imgs.csv).


#### Run script predict.py

Run the predict.py script and provide the csv path (described in #Prepare-the-images section above)

```
python3 predict.py --csv "NAME_OF_CSV"
```

#### Results

The results will be saved at "test_outputs.csv" (located in the same folder as predict.py, see sample [here](https://github.com/dchen236/FairFace/blob/master/test_outputs.csv)





The models and scripts were tested on a device with 8Gb GPU, it takes unders 2 seconds to predict the 5 images in test folder.
