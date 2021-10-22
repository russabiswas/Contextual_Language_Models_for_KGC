# Contextual Language Models for Knowledge Graph Completion

This github repository consists of the code for the paper "Contextual Language Models for Knowledge Graph Completion"

### Requirements 
  python3.5+
  
  pip install -q git+https://github.com/huggingface/transformers.git
  
  pip install -q git+https://github.com/gmihaila/ml_things.git
  
 ### Data
  
  The benchmark datasets FB13 and WN11 used in the paper is in ./data folder. The 'entity2description.txt' contains the textual entity descriptions of the entities, 'entity2text.txt' file consists of the entity name labels, and 'relation2text.txt' consists of the relation name labels, for each of the benchmark dataset.
  
  ### Instructions to run the code
  
  1. Run the 'run.sh' script in the folder ./preparedata_GPT2 to generate the data for the input to the GPT-2 model.
  2. To generate the results of triple classification using the entity labels run the code 'gpt2_fb13_label.py' for FB13, and 'gpt2_wn11_label.py' for WN11 using the commands 
  ```
  python gpt2_fb13_label.py
```
and 
```
  python gpt2_wn11_label.py
``` 
  3. To generate the results of triple classification using the entity descriptions run the code 'gpt2_fb13_desc.py' for FB13, and 'gpt2_wn11_desc.py' for WN11 using the command 
  ```
  python gpt2_fb13_desc.py 
```
and 
```
  python gpt2_wn11_desc.py
```
4. The hyper-paramters used for the models are mentioned in the paper.


