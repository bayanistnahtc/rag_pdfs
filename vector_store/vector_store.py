import logging
import pickle

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from configs import EmbeddingConfig, VectorStoreConfig, RetrievalConfig


logger = logging.getLogger(__name__)


class FAISSBMExtractRetriever:
    """
    Hybrid retriever combining FAISS and BM25.
    """

    DEFAULT_FAISS_INDEX_NAME = "faiss_index"
    DEFAULT_BM25_INDEX_NAME = "bm25_retriever.pkl"
    DEFAULT_DOCUMENTS_INDEX_NAME = "documents.pkl"
    
    def __init__(
            self,
            embedding_config: EmbeddingConfig,
            vector_config: VectorStoreConfig
        ):
        self.embedding_config = embedding_config
        self.vector_config = vector_config
        self.embeddings = None
        self.faiss_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self._documents = None

        # Initialize embeddings
        self._initialize_embeddings()

    def create_index(self, documents):
        """
        Create vector index from documents
        """
        logger.info(f"Creating vector index from {len(documents)} documents...")
        
        if not documents:
            raise ValueError("Cannot create index from empty document list")
        
        self._documents = documents

        # Create FAISS index
        logger.info("Creating FAISS index...")
        self.faiss_store = FAISS.from_documents(
            documents=documents, 
            embedding=self.embeddings
        )

        # Create BM25 retriever
        logger.info("Creating BM25 retriever...")
        # NOTE: BM25 parameters are left at default. 
        # TODO: Make parameters k1 and b configurable via config.
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        logger.info("Vector index created successfully")

    def save_index(self) -> None:
        """
        Save vector index to disk
        """
        if not self.faiss_store or not self.bm25_retriever:
            raise ValueError("No index to save. Create index first.")
        
        # Ensure directory exists
        self.vector_config.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_path = self.vector_config.index_path / "faiss_index"
        logger.info(f"Saving FAISS index to {faiss_path}")
        self.faiss_store.save_local(str(faiss_path))
        
        # Save BM25 retriever
        bm25_path = self.vector_config.index_path / "bm25_retriever.pkl"
        logger.info(f"Saving BM25 retriever to {bm25_path}")
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25_retriever, f)
        
        # Save documents for reference
        docs_path = self.vector_config.index_path / "documents.pkl"
        logger.info(f"Saving documents to {docs_path}")
        with open(docs_path, 'wb') as f:
            pickle.dump(self._documents, f)
        
        logger.info("Vector index saved successfully")
    
    def index_exists(self) -> bool:
        """
        Check if indexes exist on disk
        """
        faiss_path = self.vector_config.index_path / self.DEFAULT_FAISS_INDEX_NAME
        bm25_path = self.vector_config.index_path / self.DEFAULT_BM25_INDEX_NAME
        docs_path = self.vector_config.index_path / self.DEFAULT_DOCUMENTS_INDEX_NAME
        
        return faiss_path.exists() and \
            bm25_path.exists() and docs_path.exists() 
    
    def load_index(self) -> None:
        """
        Load vector index from disk
        """
        try:
            index_path = self.vector_config.index_path
            if not index_path.exists():
                raise FileNotFoundError(f"Index path {index_path} does not exists")
            
            faiss_path = self.vector_config.index_path / self.DEFAULT_FAISS_INDEX_NAME
            bm25_path = self.vector_config.index_path / self.DEFAULT_BM25_INDEX_NAME
            docs_path = self.vector_config.index_path / self.DEFAULT_DOCUMENTS_INDEX_NAME

            # Load FAISS index
            logger.info(f"Loading FAISS index from {faiss_path}")
            self.faiss_store = FAISS.load_local(
                faiss_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        
            # Load BM25 retriever
            logger.info(f"Loading BM25 retriever from {bm25_path}")
            with open(bm25_path, 'rb') as f:
                self.bm25_retriever = pickle.load(f)
        
            # Load documents if they exist
            if docs_path.exists():
                logger.info(f"Loading documents from {docs_path}")
                with open(docs_path, 'rb') as f:
                    self._documents = pickle.load(f)
            
            logger.info("Vector index loaded successfully")
        except Exception as e:
            logger.info(f"Failed to load indexes: {e}")

    def get_retriever(self, retrieval_config: RetrievalConfig) -> BaseRetriever:
        """
        Get ensemble retriever
        """
        if not self.faiss_store or not self.bm25_retriever:
            raise ValueError("Index not loaded. Load or create index first.")
        
        if self.ensemble_retriever is None:
            # Configure retrievers
            faiss_retriever = self.faiss_store.as_retriever(
                search_kwargs={"k": retrieval_config.k}
            )
            
            self.bm25_retriever.k = retrieval_config.k
            
            # Create ensemble retriever                
            # NOTE: The weights for the ensemble are currently hardcoded.
            # This is definitely a candidate for being moved to the config for tuning.
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, self.bm25_retriever],
                weights=[retrieval_config.faiss_weight, retrieval_config.bm25_weight]
            )

        return self.ensemble_retriever
    
    def get_index_size(self) -> int:
        """
        Return the number of documents currently indexed.

        Returns: int
            Number of indexed documents.
        """
        return len(self._documents)
    
    def _initialize_embeddings(self) -> None:
        """
        Initialize embedding model
        """
        logger.info(f"Initialize embeddings: {self.embedding_config.model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_config.model_name,
            model_kwargs=self.embedding_config.model_kwargs,
            encode_kwargs=self.embedding_config.encode_kwargs
        )

        logger.info("Embeddings initialized successfully")


class VectorStoreFactory:
    """Factory for creating vector stores"""
    
    @staticmethod
    def create_faiss_bm25_store(
        embedding_config: EmbeddingConfig,
        vector_config: VectorStoreConfig
    ) -> FAISSBMExtractRetriever:
        """
        Create FAISS + BM25 hybrid vector store
        """
        return FAISSBMExtractRetriever(embedding_config, vector_config)
