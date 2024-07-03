# From Simulations to Surrogates: Neural Networks Enhancing Burn Wound Healing Predictions

- <code>data/</code>: Contains simulation data required to run the code.
- <code>notebooks/</code>: Contains the notebooks used to train and evaluate LSTM and PINN.
- <code>scripts/</code>: Contains the scripts used to run the code for higher grid dimensions on GPU cluster.

# In order to run this code you need Python 3.8.0 and the following:

pip install tensorflow==2.10.0 <br /> 
pip install matplotlib==3.7.4 <br /> 
pip install scipy==1.10.1 <br /> 
pip install pandas==2.0.3 <br /> 
pip install deepxde==1.10.1 <br /> 
pip install numpy==1.24.3 <br /> 
pip install scikit-learn==1.3.2 <br /> 

If you want to run on GPU the compatible CUDA versions with python 3.8.0 are:

CUDA 11.2 and cuDNN 8.1
