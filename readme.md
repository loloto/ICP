# README

Thank you for reviewing our code. We have sanitized the code and prepared bash scripts with optimized hyperparameters.

## Main Files
The main scripts are as follows:
- `icp_sam.py`
- `icp_opt.py`
- `icp_llama.py`

All other files are auxiliary. 

## Notes Before Running
1. **Modify File Path Placeholders**:  
   Before running the code, you need to replace the file path placeholders in the main scripts with your personalized paths.  

2. **Pruning Tasks for OPT and Llama Models**:  
   If you intend to run the pruning tasks for OPT or Llama models, i.e., using `icp_opt.py` or `icp_llama.py`, there is no need to download additional weights or datasets manually. The code will automatically download the required weights and datasets from HuggingFace.  

3. **Pruning Tasks for SAM Models**:  
   If you plan to run pruning tasks for SAM models, you must first download the following resources:
   - SAM model weights.
   - The MS COCO 2017 dataset.
   - The `sa_000001.tar` and `sa_000003.tar` compressed files from the SA-1B dataset, which should be extracted into folders with the same names (`sa_000001` and `sa_000003`).  

   After downloading, replace the corresponding path placeholders in the script with your local paths.

## Running the Code
After modifying the file path placeholders, you can execute the code using the following commands:

```bash
bash icp_llama.sh
bash icp_opt.sh
bash icp_sam.sh
```


