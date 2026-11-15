
from vectors.chroma.store import ChromaVectorStore
from common import Config, SentenceTransformerEmbeddings

def main():

    cfg = Config()

    # Initialize embedding model
    embedding_model = SentenceTransformerEmbeddings(
        cfg.embedding_model_name
    )

    # Initialize ChromaDB
    store = ChromaVectorStore(
        persist_directory=f"{cfg.index_dir}/chroma",
        embedding_model=embedding_model,
        collection_name=cfg.collection_name,
    )

    print("\n" + "=" * 60)
    print("CHROMA DATABASE TEST")
    print("=" * 60)

    # 1. Total stored vectors
    print("\n1. TOTAL STORED VECTORS")

    print("Total:", store.count())

    # 2. Retrieve all documents
    print("\n2. ALL STORED DOCUMENTS")

    documents = store.get_all_documents()

    print("Documents retrieved:", len(documents))

    # for i, doc in enumerate(documents, start=1):

    #     print(f"\nDocument {i}")
    #     print("ID:", doc.id)
    #     print("Source:", doc.metadata.get("source"))
    #     print("Chunk:", doc.metadata.get("chunk_index"))
    #     print("Metadata:", doc.metadata)
    #     print("Content:", doc.page_content[:300])
    #     print("-" * 60)

    # 3. Retrieve one document by ID
    print("\n3. GET DOCUMENT BY ID")

    if documents:

        document_id = str(documents[0].id)

        result = store.get_document(document_id)

        if result:
            print("Found:", result.id)
            print("Content Length:", len(result.page_content[:300]))
        else:
            print("Document not found.")

    # # 4. Similarity search
    # print("\n4. SIMILARITY SEARCH")

    # query = "What is the main topic of these documents?"

    # results = store.similarity_search(
    #     query=query,
    #     k=5,
    # )

    # for i, doc in enumerate(results, start=1):

    #     print(f"\nResult {i}")
    #     print("Source:", doc.metadata.get("source"))
    #     print("Content:", doc.page_content[:300])

    # # 5. Similarity search with scores
    # print("\n5. SIMILARITY SEARCH WITH SCORES")

    # results = store.similarity_search_with_score(
    #     query=query,
    #     k=5,
    # )

    # for i, (doc, score) in enumerate(results, start=1):

    #     print(f"\nResult {i}")
    #     print("ID:", doc.id)
    #     print("Score:", score)
    #     print("Source:", doc.metadata.get("source"))
    #     print("Content:", doc.page_content[:200])

    # 6. Access underlying ChromaDB
    print("\n4. RAW CHROMA COLLECTION")

    db = store.get()

    print("Collection:", db._collection.name)
    print("Count:", db._collection.count())

    print("\n" + "=" * 60)
    print("\nDATABASE TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()