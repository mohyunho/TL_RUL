# TL_RUL
Application of Transfer Learning for RUL Prediction.
-Encode time series into 2D image.
-Merge 2D images to have one image as an input for CNN.
-Evalute widely used CNN architectures with the above data
-Apply transfer learning 


## Prerequisites
You can download the benchmark dataset used in our experiments, C-MAPSS from [here](https://drive.google.com/drive/folders/1xHLtx9laqSTO_8LOFCdOBEkouMpbkAFM?usp=sharing).
The files should be placed in /cmapss folder.
The library has the following dependencies:
```bash
pip3 install -r requirements.txt
```
-pandas <br/>
-numpy <br/>
-matplotlib <br/>
-seaborn <br/>
-sklearn <br/>
-pyts <br/>
-scikit-learn <br/>
-tensorflow-gpu <br/>


## Descriptions
- main.py: launcher for the experiments.
  - rp_creator.py
  - network.py

## Run
Please launch the experiments by 
```bash
python3 main.py -i dataset -l sequence_legnth 
```

&ndash;  i : input subdataset (1 for FD001... 4 for FD004) <br/>
&ndash;  l : sequence length of time series for each rp <br/>
--method : default='rps', help='method for encoding ts into img' <br/>
--thres_type : default='distance', help='threshold type for RPs: distance or point' <br/>
--thres_value : default=50, help='percentage of maximum distance or black points for threshold' <br/>
--n_hidden1 : default=100, help='number of neurons in the first hidden layer' <br/>
--n_hidden2 : default=10, help='number of neurons in the second hidden layer' <br/>
--epochs : default=1000, help='number epochs for network training' <br/>
--batch : default=700, help='batch size of BPTT training' <br/>

You can check all the other arguments and their details by
```bash
python3 main.py -h
```

For example,
```bash
python3 test.py -i 1 -l 30 --method rps --epochs 1000
```

After running the code, you will get the results in RMSE & Score on test data
 

## Results
Time series of the first engine (generated by run to faulure simulation) in training set
![](/figures/r2f_ts.png)

Sliced time series with fixed length (so called sequence) right before the engine failure
![](/figures/sequences.png)

RPs of the above sequences
![](/figures/rps.png)

Plot of training and validation loss in MLPs
![](/figures/loss.png)

Prediction & ground truth of RUL
![](/figures/results.png)

