# rag_next_activity

This repository contains the code and data to reproduce the experiments from the paper **.
RAG_next_activity is a conversational framework designed to predict the next activity given a trace prefix within an event log regarding the execution of a business process.
The approach leverages an architecture that combines Large Language Models (LLMs) with Retrieval Augmented Generation (RAG).

## Installation

First, you need to clone the repository:
```bash
git clone https://github.com/angelo-casciani/rag_next_activity
cd rag_next_activity
```

(Optional) Set up a conda environment for the project.
```bash
conda create -n rag_next_activity python=3.10 --yes
conda activate rag_next_activity
```

Run the following command to install the necessary dependencies using *pip*:
```bash
pip install -r requirements.txt
```

This command will read the *requirements.txt* file and install all the specified packages along with their dependencies.

Unzip the *logs.zip* directory:
```bash
unzip logs.zip
```

## LLMs Requirements

Please note that this software leverages open-source LLMs reported in the table:

| Model | HuggingFace Link |
|-----------|-----------|
| meta-llama/Meta-Llama-3.1-8B-Instruct | [HF link](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |
| meta-llama/Llama-3.2-1B-Instruct | [HF Link](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)|
| meta-llama/Llama-3.2-3B-Instruct | [HF link](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| mistralai/Mistral-7B-Instruct-v0.2 | [HF link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| mistralai/Mistral-7B-Instruct-v0.3 | [HF link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| Qwen/Qwen2.5-7B-Instruct | [HF link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| microsoft/phi-4 | [HF link](https://huggingface.co/microsoft/phi-4) |
| gpt-4o-mini | [OpenAI link](https://platform.openai.com/docs/models) |

Request in advance the permission to use each Llama model for your *HuggingFace* account.

Please note that each of the selected models have specific requirements in terms of GPU availability.
It is recommended to have access to a GPU-enabled environment meeting at least the minimum requirements for these models to run the software effectively.

## Running the Project
Before running the project, it is necessary to insert in the *.env* file:
- your personal *HuggingFace token* (request the permission to use the Llama models for this token in advance);
- the *URL* and the *gRPC port* of your *Qdrant* instance.

Eventually, you can proceed by going in the project directory and run the project in the preferred configuration. For example:
```bash
python3 src/main.py
```