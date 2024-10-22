from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import Qdrant


def initialize_vector_store(url, grpc_port, collection_name, embed_model):
    client = QdrantClient(url, grpc_port=grpc_port, prefer_grpc=True)
    store = Qdrant(client, collection_name=collection_name, embeddings=embed_model)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    return client, store


def store_prefixes(prefixes, qdrant_client, log_name, embed_model, collection_name):
    points = []
    identifier = 0
    for p in prefixes:
        p = ''.join(p)
        metadata = {'page_content': p, 'name': f'{log_name} Chunk {identifier}'}
        point = models.PointStruct(
            id=identifier,
            vector=embed_model.embed_documents([p])[0],
            payload=metadata
        )
        print(f'Processing point {identifier} of {len(prefixes)}...')
        points.append(point)
        identifier += 1

    print('Storing points into the vector store...')
    qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

    return identifier


def store_traces(traces, qdrant_client, log_name, embed_model, collection_name):
    points = []
    identifier = 0
    for t in traces:
        t = ''.join(t)
        metadata = {'page_content': t, 'name': f'{log_name} Chunk {identifier}'}
        point = models.PointStruct(
            id=identifier,
            vector=embed_model.embed_documents([t])[0],
            payload=metadata
        )
        print(f'Processing point {identifier} of {len(traces)}...')
        points.append(point)
        identifier += 1

    print('Storing points into the vector store...')
    qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

    return identifier


def retrieve_context(vector_index, query, num_chunks, key=None, search_filter=None):
    retrieved = vector_index.similarity_search(query, num_chunks)
    if key is not None and search_filter is not None:
        filter_ = models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=search_filter)
                )
            ]
        )
        meta_retrieved = vector_index.similarity_search(query, filter=filter_, k=num_chunks)
        if len(meta_retrieved) > 0:
            retrieved = meta_retrieved
    retrieved_text = ''
    for i in range(len(retrieved)):
        content = retrieved[i].page_content
        if i != len(retrieved) - 1:
            retrieved_text += f'{content}\n\n'
        else:
            retrieved_text += f'{content}'

    return retrieved_text


def delete_qdrant_collection(q_client, q_collection_name):
    q_client.delete_collection(q_collection_name)