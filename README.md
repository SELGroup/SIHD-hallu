
---

# **SIHD Project Documentation**  

## **Project Overview**  
This code repository contains all the code necessary to reproduce the experiments in the paper *"Detecting Hallucinations in Large Language Models via Structural Entropy"* (submitted to Nature Machine Intelligence). We have publicly released all the code used to generate the main experiments at the following address: [https://github.com/SELGroup/SIHD-hallu](https://github.com/SELGroup/SIHD-hallu).  

The project consists of two main modules:  
1. **Long-form text hallucination detection**  
2. **Sentence-level hallucination detection**  

---

## **Project Structure**  
```
├── README.md                      # Documentation for the project  
├── environment.yml                # Conda dependencies for the project  
├── requirements.txt               # Python dependencies  
├── long_form_structural_entropy/  # Module for long-form hallucination detection  
│   ├── DeepSeek_R1_data.py        # Handles DeepSeek-R1 dataset processing  
│   ├── HCSE.py                    # Implements Hierarchical Clustering Structural Entropy  
│   ├── main.py                    # Main entry script for long-form experiments  
│   ├── run_record/                # Stores experimental outputs  
│   └── utils.py                   # Utility functions for evaluation  
└── sentence_structural_entropy/   # Module for sentence-level hallucination detection  
    ├── analyze_results.py         # Analyzes results and metrics  
    ├── sample_answers.py          # Samples LLM-generated responses  
    ├── src/                       # Submodules for data processing  
    ├── uncertainty_quantification.py  # Implements uncertainty quantification  
    └── run_record/                # Stores experimental outputs  
```

---

## **System Requirements**

### **Hardware Dependencies**

Generally speaking, our experiments require modern computer hardware suited for working with large language models (LLMs).

The requirements for CPU and RAM are relatively modest: a system with an Intel 10th-generation CPU and 16 GB of RAM or better.

More importantly, our experiments require the use of one or more GPUs to accelerate LLM inference. Without a GPU, reproducing our results within a reasonable timeframe is not feasible. The specific GPU required depends on the size of the LLM used: larger models need GPUs with more memory.  
- **Smaller models** (7B parameters): Desktop GPUs such as the NVIDIA GeForce RTX 4090 (24 GB) are sufficient.  
- **Larger models** (13B parameters): GPUs with more memory, such as the Nvidia A100 server GPU, are required.  
- **Largest models** (70B parameters): Require the simultaneous use of two Nvidia A100 GPUs (2×80 GB) or eight NVIDIA GeForce RTX 4090 (8×24 GB).  

---

### **Software Dependencies**

Our code relies on Python 3.11 and PyTorch 2.5.1.

Our system runs on Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-89-generic x86_64).

The file [environment_export.yaml](environment_export.yaml) lists the exact versions of all Python packages used in our experiments. Please refer to the installation guide below.

---

## **Installation Guide**

To install Python and all required dependencies, we recommend using conda. For the installation guide, refer to [https://conda.io/](https://conda.io/).

After installing conda, you can set up and activate a new conda environment by executing the following commands in the shell from the root of the repository:

```bash
conda-env update -f environment.yml
conda activate SIHD
```

The installation process is expected to take approximately 20 minutes.

---

### **Setting Environment Variables**

Before running any experiments, please set the necessary environment variables to configure access to external APIs and libraries:

- **Linux/macOS**:  
  ```bash
  export HUGGING_FACE_HUB_TOKEN=<your_token>
  export OPENAI_API_KEY=<your_api_key>
  ```
  
- **Windows**:  
  ```powershell
  $env:HUGGING_FACE_HUB_TOKEN="<your_token>"
  $env:OPENAI_API_KEY="<your_api_key>"
  ```

Our experiments rely on Hugging Face for providing all LLM models and most datasets. You may need to apply for access to use the official Meta LLaMa model repository ([apply here](https://huggingface.co/meta-llama)).

Our sentence-length generation experiments use the GPT-4o model provided by the OpenAI API for accuracy assessment. Note that OpenAI charges fees based on the number of input and generated tokens. The cost of reproducing our results may vary depending on the experimental configuration, but typically ranges from $5 to $30 per run.

For most tasks, datasets are automatically downloaded upon first execution via the Hugging Face Datasets library. The only exception is BioASQ (task b, BioASQ11, 2023), which must be manually downloaded from the [download page](http://participants-area.bioasq.org/datasets). We have already downloaded and stored it at `./sentence_structural_entropy/src/data/bioasq/`.

---

## **Experimental Reproduction**  

### **1. Long-form Structural Entropy (`long_form_structural_entropy/`)**  
Detects hallucinations in long-form LLM-generated text.  

#### **Key Files**  
- `DeepSeek_R1_data.py`, `o1_data.py`: Construct fine-grained hallucination datasets generated by **DeepSeek-R1** and **OpenAI o1**.  
- `HCSE.py`: Implements **Hierarchical Clustering Structural Entropy (HCSE)** calculation.  
- `main.py`: Main entry point (LLM calls, adjacency matrix construction, structural entropy calculation, and evaluation).  
- `eval_utils.py`, `utils.py`: Evaluation and utility functions.  
- **Outputs**: Saved in `run_record/`.  

#### **Execution Steps**  
1. Install dependencies and activate the environment.  
2. Set the environment variable `OPENAI_API_KEY` to your API key to use the corresponding models.  
3. Run the main script:  
   ```bash  
   python long_form_structural_entropy/main.py  
   ```  
4. Results are saved in `long_form_structural_entropy/run_record/`.  

---

### **2. Sentence-level Structural Entropy (`sentence_structural_entropy/`)**  
Detects hallucinations at the sentence level.  

#### **Key Files**  
- `analyze_results.py`: Analyzes uncertainty metrics (e.g., AUROC, structural entropy).  
- `uncertainty_quantification.py`: Implements uncertainty quantification.  
- `sample_answers.py`: Samples answers from LLMs.  
- `src/`: Contains submodules for data processing, models, and utilities.  
- **Outputs**: Saved in `run_record/`.  

#### **Execution Steps**  
1. Install dependencies and activate the environment.  
2. Set the environment variable `OPENAI_API_KEY` to your API key to use the corresponding models.  
3. Execute the following steps:  
   - **Sample answers**:  
     ```bash  
     python sentence_structural_entropy/sample_answers.py  --model_name=$MODEL  --dataset=$DATASET $EXTRA_CFG
     ```  
   - **Run uncertainty quantification**:  
     ```bash  
     python sentence_structural_entropy/uncertainty_quantification.py --runid <run_id>  
     ```  
   - **Analyze results**:  
     ```bash  
     python sentence_structural_entropy/analyze_results.py --runid <run_id>  
     ```  
4. Results are saved in `sentence_structural_entropy/run_record/`.  
5. For detailed parameter descriptions, please refer to the `sentence_structural_entropy/src/utils/utils.py` .
---

## **Run Records**  

The `run_record/` folders in both modules store intermediate outputs (charts, metrics, etc.) for reproducibility and analysis. 

---


### **Contact**
For any questions or support, please contact the us at xtaozhao@163.com.

---

