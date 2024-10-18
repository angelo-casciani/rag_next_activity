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
                        default=10)
    parser.add_argument('--log', type=str, help='The event log to use for the next activity prediction',
                        default='Hospital_log.xes')
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate',
                        default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rebuild_vectordb', type=u.str2bool, help='Rebuild the vector index',
                        default=False)
    parser.add_argument('--modality', type=str, default='live')
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    embed_model_id = args.embed_model_id
    embed_model = p.initialize_embedding_model(embed_model_id, DEVICE, args.batch_size)

    if args.rebuild_vectordb:
        vs.delete_qdrant_collection(URL, GRPC_PORT, COLLECTION_NAME)
    q_client, q_store = vs.initialize_vector_store(URL, GRPC_PORT, COLLECTION_NAME, embed_model)
    num_docs = args.num_documents_in_context

    tree_content = lp.read_event_log(args.log)
    traces = lp.extract_traces(tree_content)
    prefixes = lp.generate_unique_prefixes(traces)
    vs.store_prefixes(prefixes, q_client, args.log, embed_model, COLLECTION_NAME)

    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
    chain = p.initialize_chain(model_id, HF_AUTH, max_new_tokens)

    run_data = {
        'Batch Size': args.batch_size,
        'Embedding Model ID': args.embed_model_id,
        'Evaluation Modality': args.modality,
        'Event Log': args.log,
        'LLM ID': args.llm_id,
        'Context Window LLM': args.model_max_length,
        'Max Generated Tokens LLM': args.max_new_tokens,
        'Number of Documents in the Context': args.num_documents_in_context,
        'Rebuilt Vector Index': args.rebuild_vectordb
    }

    # questions = {}
    if 'evaluation' in args.modality:
        pass
    else:
        p.live_prompting(model_id, chain, q_store, num_docs, run_data)


if __name__ == "__main__":
    u.seed_everything(SEED)
    main()
