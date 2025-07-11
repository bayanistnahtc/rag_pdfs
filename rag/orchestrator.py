import logging
from operator import itemgetter
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from configs import RAGConfig
from data_processing.loader_factory import DataLoaderFactory
from rag.models import Answer, DocumentChunk
from rag.prompt_templates import ANSWER_GENERATION_PROMPT


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
        Builds a RAG chain using LCEL.
        """
        # NOTE: This is a basic thread. It does not take chat history into account.
        # TODO: Integrate RunnableBranch to reformulate the question
        # based on conversation history.
        try:
            def format_documents(docs: List[Document]) -> str:
                if not docs:
                    return "Контекст не найден."
                formatted = []
                for i, doc in enumerate(docs):
                    # Предполагаем, что metadata содержит document_id и page_number
                    doc_id = doc.metadata.get('source', 'UNKNOWN')
                    page_num = doc.metadata.get('page_label', 'UNKNOWN')
                    source_tag = f"[{doc_id}/PAGE_{page_num}]"
                    formatted.append(f"Источник {i+1} {source_tag}:\n{doc.page_content}")
                return "\n\n".join(formatted)
            
            def docs_to_answer(data: Dict[str, Any]) -> Answer:
                source_chunks = []
                # for doc in data["retrieved_docs"]:
                #     chunk = DocumentChunk(
                #         content=doc.page_content,
                #         filename=doc.metadata.get('source', 'UNKNOWN'),
                #         document_id=doc.metadata.get('source', 'UNKNOWN'),
                #         page_number=doc.metadata.get('page_label', 0),
                #         chunk_index=doc.metadata.get('start_index', 0),
                #         score=doc.metadata.get('relevance_score', 0),
                #         # metadata=doc.metadata
                #     )
                #     source_chunks.append(chunk)
                
                return Answer(
                    answer_text=data["answer_text"] if data["answer_text"] else "No answer",
                    source_chunks=data["retrieved_docs"]
                )

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
