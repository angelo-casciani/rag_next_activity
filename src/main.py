from argparse import ArgumentParser
from dotenv import load_dotenv
import os
import torch
import warnings

import log_preprocessing as lp
# from oracle import AnswerVerificationOracle
import pipeline as p
import utility as u
import vector_store as vs


DEVICE = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
load_dotenv()
HF_AUTH = os.getenv('HF_TOKEN')
URL = os.getenv('QDRANT_URL')
GRPC_PORT = int(os.getenv('QDRANT_GRPC_PORT'))
COLLECTION_NAME = 'nap-rag'
SEED = 10
warnings.filterwarnings('ignore')


def parse_arguments():
    parser = ArgumentParser(description="Run LLM Generation.")
    parser.add_argument('--embed_model_id', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Embedding model identifier')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model', default=4096)
    parser.add_argument('--num_documents_in_context', type=int, help='Total number of documents in the context',
                        default=2)
    parser.add_argument('--log', type=str, help='The event log to use for the next activity prediction',
                        default='Hospital_log.xes')
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate',
                        default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rebuild_vectordb', type=u.str2bool, help='Rebuild the vector index',
                        default=False)
    parser.add_argument('--modality', type=str, default='evaluation')
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    embed_model_id = args.embed_model_id
    embed_model = p.initialize_embedding_model(embed_model_id, DEVICE, args.batch_size)

    q_client, q_store = vs.initialize_vector_store(URL, GRPC_PORT, COLLECTION_NAME, embed_model)
    num_docs = args.num_documents_in_context
    if args.rebuild_vectordb:
        vs.delete_qdrant_collection(q_client, COLLECTION_NAME)
        q_client, q_store = vs.initialize_vector_store(URL, GRPC_PORT, COLLECTION_NAME, embed_model)
        tree_content = lp.read_event_log(args.log)
        traces = lp.extract_traces(tree_content)
        # prefixes = lp.generate_unique_prefixes(traces)
        # vs.store_prefixes(prefixes, q_client, args.log, embed_model, COLLECTION_NAME)
        vs.store_traces(traces, q_client, args.log, embed_model, COLLECTION_NAME)

    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
    chain = p.initialize_chain(model_id, HF_AUTH, max_new_tokens)

    run_data = {
        'Batch Size': args.batch_size,
        'Embedding Model ID': embed_model_id,
        'Evaluation Modality': args.modality,
        'Event Log': args.log,
        'LLM ID': model_id,
        'Context Window LLM': args.model_max_length,
        'Max Generated Tokens LLM': max_new_tokens,
        'Number of Documents in the Context': num_docs,
        'Rebuilt Vector Index': args.rebuild_vectordb
    }

    # questions = {}
    if 'evaluation' in args.modality:
        prompt, answer = p.produce_answer("""<trace>
		<date key="End date:10" value="2007-01-04T23:45:36.000+01:00"/>
		<date key="Start date:10" value="2007-01-06T00:14:24.000+01:00"/>
		<int key="Diagnosis code:10" value="106"/>
		<int key="Specialism code:10" value="61"/>
		<int key="Treatment code:4" value="103"/>
		<int key="Treatment code:3" value="103"/>
		<int key="Treatment code:2" value="103"/>
		<int key="Diagnosis Treatment Combination ID:10" value="896971"/>
		<int key="Treatment code:1" value="113"/>
		<int key="Treatment code:9" value="9103"/>
		<int key="Treatment code:7" value="61"/>
		<int key="Treatment code:8" value="9103"/>
		<int key="Treatment code:5" value="9101"/>
		<int key="Treatment code:6" value="13"/>
		<int key="Treatment code:10" value="32"/>
		<date key="Start date:1" value="2005-01-25T00:14:24.000+01:00"/>
		<int key="Specialism code:3" value="61"/>
		<int key="Specialism code:4" value="61"/>
		<int key="Specialism code:5" value="13"/>
		<int key="Specialism code:6" value="61"/>
		<date key="Start date:5" value="2006-01-11T00:14:24.000+01:00"/>
		<date key="Start date:4" value="2006-01-19T00:14:24.000+01:00"/>
		<date key="Start date:3" value="2005-01-18T00:14:24.000+01:00"/>
		<int key="Specialism code:1" value="13"/>
		<int key="Specialism code:2" value="61"/>
		<date key="Start date:2" value="2005-01-18T00:14:24.000+01:00"/>
		<date key="Start date:9" value="2008-01-03T00:14:24.000+01:00"/>
		<date key="Start date:8" value="2007-01-03T00:14:24.000+01:00"/>
		<date key="Start date:7" value="2007-01-10T00:14:24.000+01:00"/>
		<date key="Start date:6" value="2006-01-10T00:14:24.000+01:00"/>
		<int key="Diagnosis code:5" value="822"/>
		<int key="Specialism code:9" value="13"/>
		<int key="Diagnosis code:4" value="106"/>
		<int key="Diagnosis code:3" value="106"/>
		<int key="Specialism code:7" value="61"/>
		<int key="Diagnosis code:2" value="106"/>
		<int key="Specialism code:8" value="13"/>
		<int key="Diagnosis code:1" value="822"/>
		<string key="Diagnosis:10" value="Gynaecologische tumoren"/>
		<int key="Diagnosis code:9" value="822"/>
		<int key="Diagnosis code:8" value="822"/>
		<int key="Diagnosis code:7" value="106"/>
		<int key="Diagnosis code:6" value="106"/>
		<string key="concept:name" value="00000012"/>
		<int key="Age:3" value="42"/>
		<int key="Age:4" value="43"/>
		<string key="Diagnosis:8" value="Maligne neoplasma cervix uteri"/>
		<string key="Diagnosis:7" value="Gynaecologische tumoren"/>
		<string key="Diagnosis:6" value="Gynaecologische tumoren"/>
		<string key="Diagnosis:5" value="Maligne neoplasma cervix uteri"/>
		<string key="Diagnosis:4" value="Gynaecologische tumoren"/>
		<int key="Age:1" value="40"/>
		<string key="Diagnosis:3" value="Gynaecologische tumoren"/>
		<int key="Age:2" value="41"/>
		<string key="Diagnosis:2" value="Gynaecologische tumoren"/>
		<string key="Diagnosis:1" value="maligniteit cervix"/>
		<string key="Diagnosis:9" value="Maligne neoplasma cervix uteri"/>
		<int key="Diagnosis Treatment Combination ID:7" value="808314"/>
		<int key="Diagnosis Treatment Combination ID:8" value="815135"/>
		<int key="Diagnosis Treatment Combination ID:9" value="890681"/>
		<int key="Diagnosis Treatment Combination ID:6" value="785993"/>
		<int key="Diagnosis Treatment Combination ID:5" value="620824"/>
		<int key="Diagnosis Treatment Combination ID:4" value="391778"/>
		<int key="Diagnosis Treatment Combination ID:3" value="389614"/>
		<int key="Diagnosis Treatment Combination ID:2" value="389613"/>
		<int key="Diagnosis Treatment Combination ID:1" value="165717"/>
		<date key="End date:8" value="2008-01-02T23:45:36.000+01:00"/>
		<date key="End date:7" value="2007-01-05T23:45:36.000+01:00"/>
		<date key="End date:1" value="2006-01-10T23:45:36.000+01:00"/>
		<date key="End date:2" value="2007-01-10T23:45:36.000+01:00"/>
		<date key="End date:3" value="2008-01-10T23:45:36.000+01:00"/>
		<date key="End date:4" value="2006-01-09T23:45:36.000+01:00"/>
		<date key="End date:5" value="2007-01-02T23:45:36.000+01:00"/>
		<date key="End date:6" value="2007-01-09T23:45:36.000+01:00"/>
		<event>
			<string key="org:group" value="Obstetrics &amp; Gynaecology clinic"/>
			<int key="Number of executions" value="1"/>
			<int key="Specialism code" value="7"/>
			<string key="concept:name" value="verlosk.-gynaec. korte kaart kosten-out"/>
			<string key="Producer code" value="SGNA"/>
			<string key="Section" value="Section 2"/>
			<int key="Activity code" value="10107"/>
			<date key="time:timestamp" value="2005-01-11T00:00:00.000+01:00"/>
			<string key="lifecycle:transition" value="complete"/>
		</event>
		<event>
			<string key="org:group" value="Pathology"/>
			<int key="Number of executions" value="1"/>
			<int key="Specialism code" value="88"/>
			<string key="concept:name" value="histologisch onderzoek - biopten nno"/>
			<string key="Producer code" value="LVPT"/>
			<string key="Section" value="Section 4"/>
			<int key="Activity code" value="356134"/>
			<date key="time:timestamp" value="2005-01-11T00:00:00.000+01:00"/>
			<string key="lifecycle:transition" value="complete"/>
		</event>""", model_id, chain, q_store, num_docs)
        print(prompt)
        print(answer)
    else:
        p.live_prompting(model_id, chain, q_store, num_docs, run_data)


if __name__ == "__main__":
    u.seed_everything(SEED)
    main()
