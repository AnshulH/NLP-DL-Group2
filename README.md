# NLP-DL-Group2
COMP_SCI 497: Deep Learning for Natural Language Processing assignments

## Details:

Current model has two models running both training and classification. 

Best trained models are final_ffnn.pt and lstm_mixed.pt, but if runnning from scratch different save files are created as checkpoints.

## Running instructions:

```

python hw1.py -model "model" -trainname "train_file_path" -testname "test_file_path" -validname "valid_file_path"


NOTE: All arg options are optional and only specifying model would work for most cases.
Also LSTM model works on .tok files so be careful while specifying paths for took files.

```


-model: All allowed arguments ["FFNN", "LSTM", "FFNN_CLASSIFY", "LSTM_CLASSIFY"]

-trainname: Correct path of the train file. Same goes for test and valid paths.
