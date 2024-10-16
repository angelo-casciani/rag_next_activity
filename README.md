# rag_next_activity

This repository contains the code and data to reproduce the experiments from the paper **.
RAG_next_activity is a conversational framework designed to predict the next activity given a trace prefix within an event log regarding the execution of a business process.
The approach leverages an architecture that combines Large Language Models (LLMs) with Retrieval Augmented Generation (RAG).

## Installation

(Optional) Set up a conda environment for the project.
```bash
conda create -n rag_next_activity python=3.10 --yes
conda activate rag_next_activity
```

To install the required Python packages for this project, you can use *pip* along with the *requirements.txt* file.

First, you need to clone the repository:
```bash
git clone https://github.com/angelo-casciani/rag_next_activity
cd rag_next_activity
```

Run the following command to install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

This command will read the requirements.txt file and install all the specified packages along with their dependencies.

## LLMs Requirements

Please note that this software leverages open-source LLMs reported in the table:

| Model | HuggingFace Link |
|-----------|-----------|
| Llama 3.1 8B Instruct | [HF link](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |

Request in advance the permission to use each Llama model for your HuggingFace account.

Please note that each of the selected models have specific requirements in terms of GPU availability.
It is recommended to have access to a GPU-enabled environment meeting at least the minimum requirements for these models to run the software effectively.

## Running the Project
Before running the project, it is necessary to insert in the *.env* file:
- your personal HuggingFace token (request the permission to use the Llama models for this token in advance);
- the URL and the gRPC port of your Qdrant instance.

Eventually, you can proceed by going in the project directory and run the project in the preferred configuration. For example:
```bash
python3 main.py
```