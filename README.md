# TL_RUL
Application of Transfer Learning for RUL Prediction.

## Prerequisites
You can download the benchmark dataset used in our experiments, C-MAPSS from [here](https://drive.google.com/drive/folders/1xHLtx9laqSTO_8LOFCdOBEkouMpbkAFM?usp=sharing).
The files should be placed in /cmapss folder.
The library has the following dependencies:
```bash
pip3 install -r requirements.txt
```
-pandas
-numpy
-matplotlib
-seaborn
-sklearn
-pyts
-scikit-learn
-tensorflow-gpu


## Descriptions
- main.py: launcher for the experiments.
  - evolutionary_algorithm.py: implementations of evolutionary algorithms to evolve neural networks in the context of predictive mainteinance.
  - task.py: implementation of a Task, used to load the data and compute the fitness of an individual.
  - utils.py: generating the multi-head CNN-LSTM network & training the network.
    - network_training.py: class for network generation and training.
    - ts_preprocessing.py: class for preprocessing and data preparation.
    - ts_window.py: class for time series window application.
- experiments.py: Evaluation of the discovered network by ENAS-PdM on unobserved data during EA & Training.

## Run
Please launch the experiments by 
```bash
python3 main.py 
```

&ndash;  i : input subdataset (1 for FD001... 4 for FD004) <br/>
&ndash;  l : sequence length of time series for each rp <br/>
--method : default='rps', help='method for encoding ts into img ' <br/>
--thres_type : default='distance', help='threshold type for RPs: distance or point ' <br/>
--thres_value : default=50, help='percentage of maximum distance or black points for threshold' <br/>
--n_hidden1 : default=100, help='number of neurons in the first hidden layer' <br/>
--n_hidden2 : default=10, help='number of neurons in the second hidden layer' <br/>
--epochs : default=1000, help='number epochs for network training' <br/>
--batch : default=700, help='batch size of BPTT training' <br/>

You can check all the other arguments and their details by
```bash
python3 main.py -h
```

 

## Results
![](/figures/r2f_ts.png)

