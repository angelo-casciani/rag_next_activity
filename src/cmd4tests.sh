#!/bin/bash

# Real-world logs
python3 src/main.py --log sepsis.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 2048
python3 src/main.py --log sepsis.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct --max_new_tokens 2048
python3 src/main.py --log sepsis.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct --max_new_tokens 2048
python3 src/main.py --log sepsis.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2 --max_new_tokens 2048
python3 src/main.py --log sepsis.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3 --max_new_tokens 2048
python3 src/main.py --log sepsis.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --max_new_tokens 2048
python3 src/main.py --log sepsis.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --max_new_tokens 2048
python3 src/main.py --log sepsis.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini --max_new_tokens 2048
python3 src/main.py --log sepsis.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max_new_tokens 32768
python3 src/main.py --log sepsis.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max_new_tokens 32768


# python3 src/main.py --log bpic20_international_declarations.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 2048
# python3 src/main.py --log bpic20_international_declarations.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct --max_new_tokens 2048
# python3 src/main.py --log bpic20_international_declarations.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct --max_new_tokens 2048
# python3 src/main.py --log bpic20_international_declarations.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2 --max_new_tokens 2048
# python3 src/main.py --log bpic20_international_declarations.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3 --max_new_tokens 2048
# python3 src/main.py --log bpic20_international_declarations.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --max_new_tokens 2048
# python3 src/main.py --log bpic20_international_declarations.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --max_new_tokens 2048
# python3 src/main.py --log bpic20_international_declarations.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini --max_new_tokens 2048

# Synthetic logs
# python3 src/main.py --log melanoma_treatment.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 2048
# python3 src/main.py --log melanoma_treatment.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct --max_new_tokens 2048
# python3 src/main.py --log melanoma_treatment.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct --max_new_tokens 2048
# python3 src/main.py --log melanoma_treatment.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2 --max_new_tokens 2048
# python3 src/main.py --log melanoma_treatment.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3 --max_new_tokens 2048
# python3 src/main.py --log melanoma_treatment.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --max_new_tokens 2048
# python3 src/main.py --log melanoma_treatment.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --max_new_tokens 2048
# python3 src/main.py --log melanoma_treatment.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini --max_new_tokens 2048

# python3 src/main.py --log udonya.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 2048
# python3 src/main.py --log udonya.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct --max_new_tokens 2048
# python3 src/main.py --log udonya.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct --max_new_tokens 2048
# python3 src/main.py --log udonya.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2 --max_new_tokens 2048
# python3 src/main.py --log udonya.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3 --max_new_tokens 2048
# python3 src/main.py --log udonya.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --max_new_tokens 2048
# python3 src/main.py --log udonya.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --max_new_tokens 2048
# python3 src/main.py --log udonya.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini --max_new_tokens 2048

# python3 src/main.py --log synthetic-13-multivariant.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 2048
# python3 src/main.py --log synthetic-13-multivariant.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct --max_new_tokens 2048
# python3 src/main.py --log synthetic-13-multivariant.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct --max_new_tokens 2048
# python3 src/main.py --log synthetic-13-multivariant.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2 --max_new_tokens 2048
# python3 src/main.py --log synthetic-13-multivariant.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3 --max_new_tokens 2048
# python3 src/main.py --log synthetic-13-multivariant.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --max_new_tokens 2048
# python3 src/main.py --log synthetic-13-multivariant.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --max_new_tokens 2048
# python3 src/main.py --log synthetic-13-multivariant.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini --max_new_tokens 2048

# python3 src/main.py --log synthetic-5-online-shopping.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 2048
# python3 src/main.py --log synthetic-5-online-shopping.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-1B-Instruct --max_new_tokens 2048
# python3 src/main.py --log synthetic-5-online-shopping.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id meta-llama/Llama-3.2-3B-Instruct --max_new_tokens 2048
# python3 src/main.py --log synthetic-5-online-shopping.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.2 --max_new_tokens 2048
# python3 src/main.py --log synthetic-5-online-shopping.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id mistralai/Mistral-7B-Instruct-v0.3 --max_new_tokens 2048
# python3 src/main.py --log synthetic-5-online-shopping.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --max_new_tokens 2048
# python3 src/main.py --log synthetic-5-online-shopping.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4 --max_new_tokens 2048
# python3 src/main.py --log synthetic-5-online-shopping.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id gpt-4o-mini --max_new_tokens 2048
