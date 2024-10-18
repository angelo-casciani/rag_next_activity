from argparse import ArgumentParser
from dotenv import load_dotenv
import warnings

import log_preprocessing as lp
#from oracle import AnswerVerificationOracle
import pipeline as p
import utility as u
import vector_store as vs


DEVICE = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
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
    parser.add_argument('--logs', nargs='+', help='The event log(s) to use for the next activity prediction',
                        default=['Hospital_log.xes'])
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rebuild_vectordb', type=str2bool, help='Rebuild the vector index', default=True)
    parser.add_argument('--modality', type=str, default='live')
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    embed_model_id = args.embed_model_id
    embed_model = p.initialize_embedding_model(embed_model_id, DEVICE, args.batch_size)

    if args.rebuild_vectordb:
        vs.delete_qdrant_collection(URL, GRPC_PORT, COLLECTION_NAME)
    qdrant = vs.initialize_vector_store(URL, GRPC_PORT, COLLECTION_NAME, embed_model)
    num_docs = args.num_documents_in_context

    lp.compute_log_stats(tree_content)
    log_name = lp.read_event_log(args.logs)
    tree_content = lp.read_event_log(log_name)
    traces = lp.extract_traces(tree_content)
    prefixes = lp.generate_prefixes(traces)
    qdrant = vs.store_prefixes(prefixes, embed_model, URL, GRPC_PORT, COLLECTION_NAME)
    



    """
    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
    chain = p.initialize_chain(model_id, HF_AUTH, max_new_tokens)

    questions = {}
    if 'evaluation' in args.modality:
        if args.modality == 'evaluation1':
            questions = load_csv_questions('1_questions_answers_not_refined_for_DFG.csv')
        elif args.modality == 'evaluation1.1':
            questions = load_csv_questions('1.1_questions_answers_refined_for_DFG.csv')
        elif args.modality == 'evaluation2':
            questions = load_csv_questions('2_questions_answers_not_refined.csv')
        elif args.modality == 'evaluation3':
            questions = load_csv_questions('3_questions_answers_refined.csv')
        elif args.modality == 'evaluation4':
            questions = load_csv_questions('4_questions_answers_different_processes.csv')
        elif args.modality == 'evaluation5':
            questions = load_csv_questions('5_questions_answers_similar_processes.csv')
        elif args.modality == 'evaluation6':
            questions = load_csv_questions('6_questions_answers_refined_ft.csv')
        oracle = AnswerVerificationOracle()
        evaluate_rag_pipeline(oracle, chain, qdrant, questions, model_id, num_docs)
    else:
        p.live_prompting(chain, qdrant, model_id, num_docs)
"""

if __name__ == "__main__":
    u.seed_everything(SEED)
    main()
