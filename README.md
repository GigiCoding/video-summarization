# Extended LSTM for Video Summarization
This code implements part of **Extended LSTM for video representation** and experiments on Charades datasets for action classification and video captioning.
## Requirements
This code is written in Lua and requires Torch. For details of prerequisites please refer to Karpathy's char-rnn.
## Usage
### Data
General information about Charades dataset can be found [here](http://allenai.org/plato/charades/).

First, download RGB features and annotation provided in Charades website:
1. [two-stream RGB features @8fps](http://ai2-website.s3.amazonaws.com/data/Charades_v1_features_rgb.tar.gz)
2. [annotation file and evaluation](http://ai2-website.s3.amazonaws.com/data/Charades.zip)

Uncompress the files and save them under a directory $DATA, which later will be specified in the option '-dir_data'. The last package should contain two ```.csv``` files: ```Charades_v1_train.csv``` and ```Charades_v1_test.csv```. These are the annotation files the code needs to import from. 
```
$ main_charades.lua -dir_data $DATA
```
### Preprocessing
Open ```./prep/trainData_gen.lua```. In line 18, change ```dir_data```, ```dir_anno```, ```dir_dest``` to your own directories. 
Repeat the same with file ```./prep/testData_gen.lua```. 

### Training
#### models:
There are four versions of our extended LSTM model: ```LSTM_extended```, ```LSTM_binaryJ```, ```LSTM_cumulative```, and ```LSTM_topK```. 

```
$ main_charades.lua -model LSTM_topK -
