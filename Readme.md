For imports to work, please use following commands on imperial machines as the dependencies are installed with appropriate versions : 

```
. /vol/cuda/10.1.105-cudnn7.6.5.32/setup.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vol/cuda/10.1.105-cudnn7.6.5.32/lib64

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vol/cuda/10.0.130/lib64

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/homes/pmk119

source /vol/bitbucket/pmk119_2/miniconda3/bin/activate
```
# To Run TVP-VAR-VDA : 
TVP-VAR-VDA for synthetic data can be run using run_tvp_tf_synthetic.py and for blockchain data using run_tvp_tf_data.py
For calling TVP-VAR-VDA for blockchain data, the signature of the function is : 
```
run_tvp_var(off_chain_file_name=file_name_off_chain, on_chain_file_name=file_name_on_chain, saveplot)
```
Example of files : 
```
file_name_off_chain = "./blockchain_data/BNB_price_diff_volume_diff.csv"
file_name_on_chain = './blockchain_data/BNB_amount_to_data_volume_diff.csv'
```
OR using command line arguments : 
```
python  run_tvp_tf_data.py --on_chain_file './blockchain_data/BNB_amount_to_data_volume_diff.csv' --off_chain_file './blockchain_data/BNB_price_diff_volume_diff.csv' --save 'temp_plot.png'
```
For synthetic data : 
```
python run_tvp_tf_synthetic.py --dim 2 --time 100 --window 1 --save 'tvp_var.png'
```

To run Kalman filter : 
```
python kalman_var.py --dim 3 --time 200 --save 'kalman.png'

```
# To Run TVP-VARNet ML model : 
TVP-VARNet ML model can be run using run_tvp_varnet.py. It needs a file consisting of beta values as input, scaling factor and position of exchange coefficient. 
Example : 
```
run_model('../TVP-VAR-VDA code/beta/BNB_amount_to_beta.csv', 10, 7)
run_model('../TVP-VAR-VDA code/beta/BNB_amount_to_y_price_x_diff_beta.csv', 1, 14)

```
Via command line arguments : 
```
python run_tvp_varnet.py --beta_file '../TVP-VAR-VDA code/beta/BNB_amount_to_beta.csv' --scaling_factor 10 --beta_plot_pos 7 --save 'test2.png'
```
# Data
Blockchain data exists in folder called blockchain_data and beta files exist in folder called beta. Blockchain data has been generated using python notebooks :
BNB Generate On-chain and Off-chain data.ipynb, Data On-Chain ERC20 data.ipynb
