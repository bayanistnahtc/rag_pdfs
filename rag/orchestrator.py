import logging
import time
from operator import itemgetter
from typing import Any, Dict, List

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from configs import RAGConfig
from data_processing.loader_factory import DataLoaderFactory
from rag.models import Answer, DocumentChunk
from rag.prompt_templates import ANSWER_GENERATION_PROMPT
from rag.utils import format_documents, docs_to_answer

from llm.client import LLMFactory
from vector_store.vector_store import VectorStoreFactory


logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """The orchestrator of the main rag pipeline"""
    def __init__(
            self, 
            config: RAGConfig
        ):
        self.config = config
        self.data_loader = None
        self.vector_store = None
        self.retriever = None
        self.compressor = None
        self.llm = None
        self._rag_chain = None
    
    def initialize(
            self,
            force_rebuild_storage: bool = False
    ) -> None:
        """
        Initialize all RAG components

        Args: 
            force_rebuild_storage: Whether to force rebuild of indexes
        
        Raises:
            Exception: If initialization failed
        """

        try:
            logger.info("Initializing RAG pipeline...")

            # Initialize data loader
            self._initialize_data_loader()
            

            # Initialize vector store
            self.vector_store = VectorStoreFactory.create_faiss_bm25_store(
                embedding_config=self.config.embedding,
                vector_config=self.config.vector_store
            )

            # Load or create indexes
            if force_rebuild_storage or not self.vector_store.index_exists():
                logger.info("Creating new indexes...")
                self._build_indexes()
            else:
                logger.info("Loading existing indexes...")
                self.vector_store.load_index()

            # Initialize trtriever
            self.retriever = self.vector_store.get_retriever(self.config.retrieval)

            # Initialize LLM
            llm_client = LLMFactory.create_llm(self.config.llm)
            self.llm = llm_client.get_langchain_llm()

            # Build RAG chain
            self._build_chain()
            logger.info("RAG pipeline initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise

    def _build_indexes(self) -> None:
        """
        Build vector indexes from data

        Raises:
            ValueError: If no documents loaded
            Exception: If index building fails
        """
        try:
            # Load documents
            logger.info("Loading documents...")
            documents = self.data_loader.load_documents(use_local=False)

            if not documents:
                raise ValueError("No documents loaded")
            
            # Process documents
            logger.info("Processing documents...")
            processed_documents = self.data_loader.process_documents(documents)
            # chuncked_documents = self.text_splitter.split_documents(processed_documents)

            # Create indexes
            logger.info("Creating vector indexes...")
            self.vector_store.create_index(processed_documents)

            # Save indexes
            logger.info("Saving indexes to disk...")
            self.vector_store.save_index()

            logger.info(f"Successfully build indexes from {len(processed_documents)} document chunks")

        except Exception as e:
            logger.error(f"Failed to build indexes: {e}")
            raise

    def _initialize_data_loader(self) -> None:
        """
        Initialize data loade component.
        """
        try:
            self.data_loader = DataLoaderFactory.create_documents_loader(
                documents_config=self.config.documents_config,
                chunking_config=self.config.text_splitter,
                vector_store_config=self.config.vector_store,
                
            )
        except Exception as e:
            logger.error(f"Failed to initialize data loader: {e}")
            raise

    def _build_chain(self):
        """
        Build RAG chain using LangChain Expression Language (LCEL).
         
        Raises:
            RuntimeError: If chain building fails
            ValueError: If required components are missing
            """
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        if not self.llm:
            raise ValueError("LLM not initialized")

        # NOTE: This is a basic thread. It does not take chat history into account.
        # TODO: Integrate RunnableBranch to reformulate the question
        # based on conversation history.
        try:
            self._rag_chain = (
                # step 1: retrieve chunks
                RunnablePassthrough()
                | {
                    "question": itemgetter("question"),
                    "retrieved_docs": itemgetter("question") | self.retriever,
                }
                # step 2: chunks formatting
                | RunnableLambda(lambda x: {
                    "question": x["question"],
                    "context": format_documents(x["retrieved_docs"]),
                    "retrieved_docs": x["retrieved_docs"],
                })
                # step 3: generate anser
                | {
                    "answer_text": ANSWER_GENERATION_PROMPT | self.llm | StrOutputParser(),
                    "retrieved_docs": itemgetter("retrieved_docs"),
                }
                # Шаг 4: Forming the final Answer object
                | RunnableLambda(docs_to_answer)
            )
            logger.error("RAG chain build successfully")
        except Exception as e:
            logger.error(f"Failed to build RAG chain: {e}")
            raise

    def invoke(self, query: str) -> Answer:
        # HACK: Workaround for output transformation.
        # Need to create a Pydantic model for the response and a parser.
        result = self._rag_chain.invoke({"question": query, "chat_history": ""})
        return result


    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform comprehensive health check of all RAG components.
        
        Returns: Dict[str, Dict[str, Any]]
            Dict mapping component names to their health status
        """
        components = {}
        
        # Check vector store
        try:
            if self.vector_store and self.vector_store.index_exists():
                components["vector_store"] = {
                    "status": "healthy",
                    "index_size": self.vector_store.get_index_size(),
                }
            else:
                components["vector_store"] = {"status": "unhealthy", "error": "Index not found"}
        except Exception as e:
            components["vector_store"] = {"status": "unhealthy", "error": str(e)}
        
        # Check LLM
        try:
            if self.llm:
                # Quick health check with simple query
                start_time = time.time()
                test_response = self.llm.invoke("Test")
                latency = (time.time() - start_time) * 1000
                components["llm"] = {
                    "status": "healthy" if test_response else "degraded",
                    "latency_ms": round(latency, 2)
                }
            else:
                components["llm"] = {"status": "unhealthy", "error": "LLM not initialized"}
        except Exception as e:
            components["llm"] = {"status": "unhealthy", "error": str(e)}
        
        # Check retriever
        try:
            if self.retriever:
                # Test retrieval with simple query
                test_docs = self.retriever.get_relevant_documents("test", k=1)
                components["retriever"] = {
                    "status": "healthy",
                    "test_results": len(test_docs)
                }
            else:
                components["retriever"] = {"status": "unhealthy", "error": "Retriever not initialized"}
        except Exception as e:
            components["retriever"] = {"status": "unhealthy", "error": str(e)}
        
        return components
    