# video-summarization
This code implements **Extended LSTM for video representation** and experiments on Charades datasets for action classification and video captioning.
## Requirements:
This code is written in Lua and requires Torch. For details of prerequisites please refer to Karpathy's char-rnn.
## Usage:
### Data
General information about Charades dataset can be found here: http://allenai.org/plato/charades/
To run this code, first downloan the following items:
(1) two-stream RGB features @8fps here: http://ai2-website.s3.amazonaws.com/data/Charades_v1_features_rgb.tar.gz
(2) annotation file and evaluation here: http://ai2-website.s3.amazonaws.com/data/Charades.zip
Uncompress those files and save them under a directory
In the folder where you download, call 
```
$ main_charades.lua -model LSTM_topK -l1_weight 0.01 
```

