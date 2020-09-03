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
run_tvp_var(off_chain_file_name=file_name_off_chain, on_chain_file_name=file_name_on_chain)
```
Example of files : 
```
file_name_off_chain = "./blockchain_data/BNB_ammount_to_price_diff_volume_diff.csv"
file_name_on_chain = './blockchain_data/BNB_amount_to_data_volume_diff.csv'
```
# To Run TVP-VARNet ML model : 
TVP-VARNet ML model can be run using run_tvp_varnet.py. It needs a file consisting of beta values as input, scaling factor and position of exchange coefficient. 
Example : 
```
run_model('../TVP-VAR-VDA code/beta/BNB_amount_to_beta.csv', 10, 7)
run_model('../TVP-VAR-VDA code/beta/BNB_amount_to_y_price_x_diff_beta.csv', 1, 14)

```
# Data
blockchain data exists in folder called blockchain_data and beta files exist in folder called beta. Blockchain data has been generated using python notebooks :
BNB Generate On-chain and Off-chain data.ipynb, Data On-Chain ERC20 data.ipynb