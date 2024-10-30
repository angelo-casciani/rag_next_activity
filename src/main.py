from argparse import ArgumentParser
from dotenv import load_dotenv
import os
import torch
import warnings

import log_preprocessing as lp
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

# 'sentence-transformers/all-MiniLM-L6-v2'
# 'sentence-transformers/all-mpnet-base-v2'
def parse_arguments():
    parser = ArgumentParser(description="Run LLM Generation.")
    parser.add_argument('--embed_model_id', type=str, default='sentence-transformers/all-MiniLM-L12-v2',
                        help='Embedding model identifier')
    parser.add_argument('--vector_dimension', type=int, default=384,
                        help='Vector space dimension')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model',
                        default=128000)
    parser.add_argument('--num_documents_in_context', type=int, help='Number of documents in the context',
                        default=10)
    parser.add_argument('--log', type=str, help='The event log to use for the next activity prediction',
                        default='Hospital_log.xes')
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate',
                        default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rebuild_db_and_tests', type=u.str2bool,
                        help='Rebuild the vector index and the test set', default=True)
    parser.add_argument('--modality', type=str, default='evaluation')
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    embed_model_id = args.embed_model_id
    embed_model = p.initialize_embedding_model(embed_model_id, DEVICE, args.batch_size)
    space_dimension = args.vector_dimension

    q_client, q_store = vs.initialize_vector_store(URL, GRPC_PORT, COLLECTION_NAME, embed_model, space_dimension)
    num_docs = args.num_documents_in_context
    test_set_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'validation',
                             f"test_set_{args.log.split('.xes')[0]}.csv")
    if args.rebuild_db_and_tests:
        vs.delete_qdrant_collection(q_client, COLLECTION_NAME)
        q_client, q_store = vs.initialize_vector_store(URL, GRPC_PORT, COLLECTION_NAME, embed_model, space_dimension)
        """
        tree_content = lp.read_event_log(args.log)
        traces_list = lp.extract_traces(tree_content)
        prefixes = lp.generate_prefix_windows(traces_list)
        vs.store_prefixes(prefixes, q_client, args.log, embed_model, COLLECTION_NAME)
        """
        # vs.store_traces(traces, q_client, args.log, embed_model, COLLECTION_NAME)
        content = lp.read_event_log(args.log)
        traces = lp.extract_traces_concept_names(content)
        test_set = lp.generate_test_set(traces, 0.2)
        traces_retrieval = [trace for trace in traces if trace not in test_set]
        lp.generate_csv_from_test_set(test_set, test_set_path)
        vs.store_traces_concept_names(traces_retrieval, q_client, args.log, embed_model, COLLECTION_NAME)

    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
    chain = p.initialize_chain(model_id, HF_AUTH, max_new_tokens)

    run_data = {
        'Batch Size': args.batch_size,
        'Embedding Model ID': embed_model_id,
        'Vector Space Dimension': space_dimension,
        'Evaluation Modality': args.modality,
        'Event Log': args.log,
        'LLM ID': model_id,
        'Context Window LLM': args.model_max_length,
        'Max Generated Tokens LLM': max_new_tokens,
        'Number of Documents in the Context': num_docs,
        'Rebuilt Vector Index and Test Set': args.rebuild_db_and_tests
    }

    if 'evaluation' in args.modality:
        test_dict = u.load_csv_questions(test_set_path)
        p.evaluate_rag_pipeline(model_id, chain, q_store, num_docs, test_dict, run_data)
        """prompt, answer = p.produce_answer("verlosk.-gynaec. korte kaart kosten-out, histologisch onderzoek - biopten nno, ",
                                          model_id, chain, q_store, num_docs)
        print(prompt)
        print(answer)"""
    else:
        p.live_prompting(model_id, chain, q_store, num_docs, run_data)


if __name__ == "__main__":
    u.seed_everything(SEED)
    main()
