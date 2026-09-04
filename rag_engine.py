import chromadb
from chromadb.utils import embedding_functions

# Use a lightweight local embedding model
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# In-memory database for session data; persistent directory for global data
chroma_client = chromadb.Client()
persistent_client = chromadb.PersistentClient(path="./global_rag_db")

session_collection = chroma_client.create_collection(
    name="session_notes", embedding_function=embedding_fn
)
global_collection = persistent_client.get_or_create_collection(
    name="global_knowledge", embedding_function=embedding_fn
)

def index_session_note(note_id: str, text: str, metadata: dict = None):
    session_collection.add(
        ids=[note_id],
        documents=[text],
        metadatas=[metadata or {}]
    )

def index_global_doc(doc_id: str, text: str, metadata: dict = None):
    global_collection.add(
        ids=[doc_id],
        documents=[text],
        metadatas=[metadata or {}]
    )

def retrieve_relevant_context(query: str, threshold: float = 0.8, n_results: int = 3) -> tuple[list[str], str]:
    # Search local session database first
    local_res = session_collection.query(query_texts=[query], n_results=n_results)
    
    local_docs = local_res.get("documents", [[]])[0]
    local_distances = local_res.get("distances", [[]])[0]
    
    # Check if local top result meets similarity threshold
    if local_docs and local_distances and local_distances[0] <= threshold:
        return local_docs, "local"

    # Fallback: search global database
    global_res = global_collection.query(query_texts=[query], n_results=n_results)
    global_docs = global_res.get("documents", [[]])[0]
    
    if global_docs:
        return global_docs, "global"
        
    # If neither yields matches, fall back to local docs if available
    return local_docs if local_docs else [], "none"
