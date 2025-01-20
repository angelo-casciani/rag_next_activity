#!/bin/bash

# Evaluation of Llama 3.1 Instruct on Road Traffic Fine Management Log ONLY concept names
python3 src/main.py --log Road_Traffic_Fine_Management_Process.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --num_documents_in_context 3 --max_new_tokens 1536
python3 src/main.py --log Hospital_billing.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --num_documents_in_context 3 --max_new_tokens 1536
python3 src/main.py --log BPI_Challenge_2012.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --num_documents_in_context 3 --max_new_tokens 1536
python3 src/main.py --log BPI_Challenge_2013_incidents.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --num_documents_in_context 3 --max_new_tokens 1536
python3 src/main.py --log BPIC15_1.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --num_documents_in_context 3 --max_new_tokens 1536
python3 src/main.py --log BPI_Challenge_2017.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --num_documents_in_context 3 --max_new_tokens 1536
python3 src/main.py --log sintetico-5-online-shopping-alt.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --num_documents_in_context 3 --max_new_tokens 1536

# python3 src/main.py --log sepsis.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --num_documents_in_context 3 --max_new_tokens 1536




# python3 src/main.py --log Road_Traffic_Fine_Management_Process.xes --modality evaluation-concept_names --rebuild_db_and_tests True --log_gap 3
# Evaluation of Llama 3.1 Instruct on Hospital Log ONLY concept names
# python3 src/main.py --log Hospital_log.xes --modality evaluation-concept_names --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on Hospital Billing Log ONLY concept names
# python3 src/main.py --log Hospital_billing.xes --modality evaluation-concept_names --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-1-1 Log ONLY concept names
# python3 src/main.py --log sintetico-1-1var-relevant.xes --modality evaluation-concept_names --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-2-2 Log ONLY concept names
# python3 src/main.py --log sintetico-2-2var-1rel-1-nonrel.xes --modality evaluation-concept_names --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-3-2 Log ONLY concept names
# python3 src/main.py --log sintetico-3-2var-2rel.xes --modality evaluation-concept_names --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-5 Log ONLY concept names
# python3 src/main.py --log sintetico-5-online-shopping.xes --modality evaluation-concept_names --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-5-alt Log ONLY concept names
# python3 src/main.py --log sintetico-5-online-shopping-alt.xes --modality evaluation-concept_names --rebuild_db_and_tests True


# Evaluation of Llama 3.1 Instruct on Road Traffic Fine Management Log with log attributes
# python3 src/main.py --log Road_Traffic_Fine_Management_Process.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
# Evaluation of Llama 3.1 Instruct on Hospital Log with log attributes
# python3 src/main.py --log Hospital_log.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
# Evaluation of Llama 3.1 Instruct on Hospital Billing Log with log attributes
# python3 src/main.py --log Hospital_billing.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
# Evaluation of Llama 3.1 Instruct on sintetico-1-1 Log with log attributes
# python3 src/main.py --log sintetico-1-1var-relevant.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
# Evaluation of Llama 3.1 Instruct on sintetico-2-2 Log with log attributes
# python3 src/main.py --log sintetico-2-2var-1rel-1-nonrel.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
# Evaluation of Llama 3.1 Instruct on sintetico-3-2 Log with log attributes
# python3 src/main.py --log sintetico-3-2var-2rel.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
# Evaluation of Llama 3.1 Instruct on sintetico-5 Log with log attributes
# python3 src/main.py --log sintetico-5-online-shopping.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
# Evaluation of Llama 3.1 Instruct on sintetico-5-alt Log with log attributes
# python3 src/main.py --log sintetico-5-online-shopping-alt.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct

# python3 src/main.py --log sepsis.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
# python3 src/main.py --log BPI_Challenge_2017.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
# python3 src/main.py --log BPI_Challenge_2012.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
# python3 src/main.py --log BPIC15_1.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct

# python3 src/main.py --log sintetico-11-insurance_nodata.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct
# python3 src/main.py --log sintetico-12-nodata.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct

python3 src/main.py --log sintetico-11-insurance_data_norel.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --num_documents_in_context 3 --max_new_tokens 1536
python3 src/main.py --log sintetico-12-reserveroom_data_norel.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct --num_documents_in_context 3 --max_new_tokens 1536
python3 src/main.py --log sintetico-13-multivariant.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id Qwen/Qwen2.5-7B-Instruct

# python3 src/main.py --log BPI_Challenge_2013_incidents.xes --modality evaluation-attributes --rebuild_db_and_tests True --llm_id microsoft/phi-4
