#!/bin/bash

# Evaluation of Llama 3.1 Instruct on Hospital Log ONLY concept names
python3 src/main.py --log Hospital_log.xes --modality evaluation-only-concept_names --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on Road Traffic Fine Management Log ONLY concept names
python3 src/main.py --log Road_Traffic_Fine_Management_Process.xes --modality evaluation-only-concept_names --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on Hospital Billing Log ONLY concept names
python3 src/main.py --log Hospital_billing.xes --modality evaluation-only-concept_names --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-1-1 Log ONLY concept names
python3 src/main.py --log sintetico-1-1var-relevant.xes --modality evaluation-only-concept_names --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-2-2 Log ONLY concept names
python3 src/main.py --log sintetico-2-2var-1rel-1-nonrel.xes --modality evaluation-only-concept_names --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-3-2 Log ONLY concept names
python3 src/main.py --log sintetico-3-2var-2rel.xes --modality evaluation-only-concept_names --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-5 Log ONLY concept names
python3 src/main.py --log sintetico-5-online-shopping.xes --modality evaluation-only-concept_names --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-5-alt Log ONLY concept names
python3 src/main.py --log sintetico-5-online-shopping-alt.xes --modality evaluation-only-concept_names --rebuild_db_and_tests True


# Evaluation of Llama 3.1 Instruct on Hospital Log with log attributes
python3 src/main.py --log Hospital_log.xes --modality evaluation-attributes --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on Road Traffic Fine Management Log with log attributes
python3 src/main.py --log Road_Traffic_Fine_Management_Process.xes --modality evaluation-attributes --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on Hospital Billing Log with log attributes
python3 src/main.py --log Hospital_billing.xes --modality evaluation-attributes --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-1-1 Log with log attributes
python3 src/main.py --log sintetico-1-1var-relevant.xes --modality evaluation-attributes --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-2-2 Log with log attributes
python3 src/main.py --log sintetico-2-2var-1rel-1-nonrel.xes --modality evaluation-attributes --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-3-2 Log with log attributes
python3 src/main.py --log sintetico-3-2var-2rel.xes --modality evaluation-attributes --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-5 Log with log attributes
python3 src/main.py --log sintetico-5-online-shopping.xes --modality evaluation-attributes --rebuild_db_and_tests True
# Evaluation of Llama 3.1 Instruct on sintetico-5-alt Log with log attributes
python3 src/main.py --log sintetico-5-online-shopping-alt.xes --modality evaluation-attributes --rebuild_db_and_tests True
