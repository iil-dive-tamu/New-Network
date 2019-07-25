

# New-Network

## Introduction
This is the prilimary implimation of the idea from Professor Anxiao Jiang. you can read the `New_Network_Architecture.pdf` for more information.

## 

## Structure
`train.py` is the training code and testing code have not implemented yet since there are some problems need to be solved. The folder `utils` including all the files we need. You can use `requirement.txt` to install all the packages we need to run this network. 

## Training
I use `optparse` as the parser of the `train.py`. Only three parameters are necessary and all of others have default value. 
These three parameters are the location of the original image, initial probaility map and label.
```
Sample:
python train.py --image "C:\Users\sunzh\CS636\Summer project\BPN\data\train-input\train-input.tif"  --promap "C:\Users\sunzh\CS636\Summer project\BPN\data\PROMAP\train-promap.tif"  --label "C:\Users\sunzh\CS636\Summer project\BPN\data\train-labels\train-labels.tif" --path 'C:\Users\sunzh\CS636\Summer project\BPN\data'
```
The folder of `path` should have three subfolder named with `FOV` `IMAGE` `LABEL`
