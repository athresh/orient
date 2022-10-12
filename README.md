# ORIENT

This is the implementation of the paper "ORIENT: Submodular Mutual Information Measures for Data Subset Selection under Distribution Shift." 

The main runner file is `run_sl_smi.py`.

```
usage: run_sl_smi.py [-h] [--config_file CONFIG_FILE] [--smi_func_type SMI_FUNC_TYPE] [--query_size QUERY_SIZE]
                     [--fraction FRACTION] [--select_every SELECT_EVERY] [--print_every PRINT_EVERY]
                     [--save_every SAVE_EVERY] [--device DEVICE] [--num_epochs NUM_EPOCHS]
                     [--source_domains SOURCE_DOMAINS] [--target_domains TARGET_DOMAINS]
                     [--similarity_criterion SIMILARITY_CRITERION] [--selection_type SELECTION_TYPE]

optional arguments:
  -h, --help            show this help message and exit
  --config_file         path to the config file 
  --smi_func_type       SMI function to be used, options ["fl2mi", "gcmi", "logdetmi"]
  --query_size          size of the query set i.e. target data 
  --fraction            fraction of the source data to be used in the subset
  --select_every        subset selection interval
  --print_every         interval for evaluating and printing the performance on target data
  --save_every          interval for saving the model checkpoint
  --device              cpu/gpu
  --num_epochs          total number of epochs
  --source_domains      source domain, for eg. Real_World
  --target_domains      target domain, for eg. Clipart
  --similarity_criterion  Criterion for similarity (use default)
  --selection_type      Selection type     
```

Some sample commands to run the code for CCSA on office-home dataset are mentioned below. Similar commands can be used to run d-sne setting.


1. For Full
```shell
python run_sl_smi.py --config_file "configs/SL/config_full_ccsa_officehome.py" --source_domains "Real_World" --target_domains "Clipart" > full_ccsa_officehome_rc.txt
```
2. Random

```shell
python run_sl_smi.py --config_file "configs/SL/config_random_ccsa_officehome.py" --fraction "0.3"  --query_size 624 --source_domains "Real_World" --target_domains"Clipart" > random_ccsa_0.3_officehome_rc.txt
```

3. ORIENT (FLMI)

```shell
python run_sl_smi.py --config_file "configs/SL/config_smi_ccsa_officehome.py" --smi_func_type "fl2mi"  --fraction "0.3"  --query_size 624 --source_domains "Real_World"--target_domains "Clipart" > fl2mi_ccsa_0.3_officehome_rc.txt
```

4. GLISTER

```shell
python run_sl_smi.py --config_file "configs/SL/config_glister_ccsa_officehome.py" --fraction "0.3"  --query_size 624 --source_domains "Real_World" --target_domains"Clipart" > glister_ccsa_0.3_officehome_rc.txt
```   

5. GradMatch

```shell
python run_sl_smi.py --config_file "configs/SL/config_gradmatch_ccsa_officehome.py" --fraction "0.3"  --query_size 624 --source_domains "Real_World" --target_domains "Clipart" --selection_type 'PerClassPerGradient'> gradmatch_ccsa_0.3_officehome_rc.txt
```




## Citation

If you build on this code or the ideas of this paper, please use the following citation.

    @inproceedings{KaranamKKI22,
     	title={ORIENT: Submodular Mutual Information Measures for Data Subset Selection under Distribution Shift}, 
	    author={Athresh Karanam and Krishnateja Killamsetty and Harsha Kokel and Rishabh K Iyer}, 
    	year={2022}, 
		booktitle={NeurIPS},
    }


## Acknowledgements

AK acknowledges the support by the NIH grant R01HD101246, HK gratefully acknowledges the support of the ARO award W911NF2010224. RI and KK would like to acknowledge support from NSF Grant Number IIS-2106937, a gift from Google Research, and the Adobe Data Science Research award. Authors would like to acknowledge Dr. Sriraam Natarajan for helpful discussions and support. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the ARO, NIH, NSF, Google Research, Adobe Data Science or the U.S. government.
