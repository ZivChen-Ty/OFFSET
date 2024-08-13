# OFFSET: Segmentation-based Focus Shift Revision for Composed Image Retrieval

This is an open-source implementation of the paper "OFFSET: Segmentation-based Focus Shift Revision for Composed Image Retrieval" (**OFFSET**).
*Ckpt&Data: The checkpoints and dominant portion segmentation data of OFFSET is available in xxx.*

### Installation
1. Clone the repository

```sh
git clone https://anonymous.4open.science/r/OFFSET
```

2. Running Environment

```sh
Platform: NVIDIA RTX 3090 24G
Python  3.8.10
Pytorch  2.0.0
```


### Data Preparation

#### Shoes

Download the Shoes dataset following the instructions in
the [**official repository**](https://github.com/XiaoxiaoGuo/fashion-retrieval/tree/master/dataset).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── Shoes
│   ├── captions_shoes.json
│   ├── eval_im_names.txt
│   ├── relative_captions_shoes.json
│   ├── train_im_names.txt
│   ├── [womens_athletic_shoes | womens_boots | ...]
|   |   ├── [0 | 1]
|   |   ├── [img_womens_athletic_shoes_375.jpg | descr_womens_athletic_shoes_734.txt | ...]

```

#### FashionIQ

Download the FashionIQ dataset following the instructions in
the [**official repository**](https://github.com/XiaoxiaoGuo/fashion-iq).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── FashionIQ
│   ├── captions
|   |   ├── cap.dress.[train | val | test].json
|   |   ├── cap.toptee.[train | val | test].json
|   |   ├── cap.shirt.[train | val | test].json

│   ├── image_splits
|   |   ├── split.dress.[train | val | test].json
|   |   ├── split.toptee.[train | val | test].json
|   |   ├── split.shirt.[train | val | test].json

│   ├── dress
|   |   ├── [B000ALGQSY.jpg | B000AY2892.jpg | B000AYI3L4.jpg |...]

│   ├── shirt
|   |   ├── [B00006M009.jpg | B00006M00B.jpg | B00006M6IH.jpg | ...]

│   ├── toptee
|   |   ├── [B0000DZQD6.jpg | B000A33FTU.jpg | B000AS2OVA.jpg | ...]
```

#### Fashion200K

Download the Fashion200K dataset following the instructions in
the [**official repository**](https://github.com/xthan/fashion-200k.git).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── Fashion200K
│   ├── test_queries.txt

│   ├── labels
|   |   ├── dress_[train | test]_detect_all.txt
|   |   ├── jacket_[train | test]_detect_all.txt
|   |   ├── pants_[train | test]_detect_all.txt
|   |   ├── skirt_[train | test]_detect_all.txt
|   |   ├── top_[train | test]_detect_all.txt

│   ├── women
|   |   ├── [dresses | jackets | pants | skirts | tops]
```


#### CIRR

Download the CIRR dataset following the instructions in the [**official repository**](https://github.com/Cuberick-Orion/CIRR).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── CIRR
│   ├── train
|   |   ├── [0 | 1 | 2 | ...]
|   |   |   ├── [train-10108-0-img0.png | train-10108-0-img1.png | ...]

│   ├── dev
|   |   ├── [dev-0-0-img0.png | dev-0-0-img1.png | ...]

│   ├── test1
|   |   ├── [test1-0-0-img0.png | test1-0-0-img1.png | ...]

│   ├── cirr
|   |   ├── captions
|   |   |   ├── cap.rc2.[train | val | test1].json
|   |   ├── image_splits
|   |   |   ├── split.rc2.[train | val | test1].json
```

#### Train

Train OFFSET on Shoes, FashionIQ, Fashion200K, CIRR.

```sh
python3 train.py 
--model_dir ... 
--dataset {shoes, fashioniq, cirr, fashion200k}
--cirr_path ""
--fashioniq_path ""
--shoes_path ""
--fashion200k_path ""
```

```
--dataset <str>                 Dataset to use, options: ['fashioniq', 'shoes', 'cirr', 'fashion200k']
--cirr_path <str>               Path to the CIRR dataset root folder
--fashioniq_path <str>          Path to the FashionIQ dataset root folder
--shoes_path <str>              Path to the Shoes dataset root folder
--fashion200k_path <str>        Path to the Fashion200K dataset root folder
--model_dir <str>               Path to save checkpoints and logs
```


</details>


### Inference Phase

#### Validation

Evaluate OFFSET on Shoes, FashionIQ, Fashion200K, CIRR.

```sh
python3 evaluation.py 
--model_dir checkpoints/OFFSET_{Shoes,FashionIQ,CIRR, Fashion200K}.pth 
--dataset {shoes, fashioniq, cirr, fashion200k}
--cirr_path ""
--fashioniq_path ""
--shoes_path ""
--fashion200k_path ""
```

```
--dataset <str>                 Dataset to use, options: ['fashioniq', 'shoes', 'cirr', 'fashion200k']
--cirr_path <str>               Path to the CIRR dataset root folder
--fashioniq_path <str>          Path to the FashionIQ dataset root folder
--shoes_path <str>              Path to the Shoes dataset root folder
--fashion200k_path <str>        Path to the Fashion200K dataset root folder
--model_dir <str>               Path of the pre-trained model
```


</details>


#### Test for CIRR

To generate the predictions file for uploading on the [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/) using the our model, please execute the following command:

```sh
python src/cirr_test_submission.py model_path
```

```
model_path <str> : Path of the OFFSET checkpoint on CIRR, e.g. "checkpoints/OFFSET_CIRR.pt"
```


</details>



### Acknowledgement


