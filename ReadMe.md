# Deep Learning Project

A deep learning project to learn NLP methods and LSTM neural networks. Made as part of the course DD2424: Deep Learning in Data Science at KTH, Stockholm.
We uppgaded the RNN netowrk and build LSTM network, both one layer and two layer

## Quickstart
### Pre-requisites
* Program tested on Python 3.11 and 3.12.

### Install
Quick install:
```
git clone https://github.com/AntoineMissue/DD2424-project.git
pip install -r requirements.txt
```

Recommended install (virtual env)
* macOS/Linux
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

* Windows
```
python -m venv venv
.\venv\Scripts\activate.bat
.\venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

### Training the RNN model
To train the base RNN, adjust the training parameters (batch size, number of epochs, optimizer between Adam and AdaGrad) in the `src/nlpProject/rnn_baseline.py` file. It is also possible to modify the name of the training curve figure and the path to the saved model. Then, run:   
```
python src/nlpProject/rnn_baseline.py
```
To save the training info:
```
python src/nlpProject/rnn_baseline.py > ./reports/logs/training_info.txt
```
The training curve will be saved in `src/reports/figures/` and the trained model will be saved in `models/RNN`.

### Generating text with the RNN model
To generate text with the RNN model, adjust generation parameters and specify a saved model filename in `src/nlpProject/inference.py`. Then, run:
```
python src/nlpProject/inference.py
```
To save the output:
```
python src/nlpProject/inference.py > ./reports/logs/generated_text.txt
```
