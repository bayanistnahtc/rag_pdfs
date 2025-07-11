import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs: Dict[str, Any] = field(default_factory=lambda: {"device": "cpu"})
    encode_kwargs: Dict[str, Any] = field(default_factory=lambda: {"normalize_embeddings": True})
    
    def post_init(self):
        if self.model_kwargs is None:
            self.model_kwargs = {"device": "cpu"}
        if self.encode_kwargs is None:
            self.encode_kwargs = {"normalize_embeddings": True}


@dataclass
class VectorStoreConfig:
    """Configuration for vector store operations."""
    index_path: Path = Path("data/indexes")
    chunk_size: int = 1000
    chunk_overlap: int = 200

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval."""
    k: int = 10
    bm25_weight: float = 0.5
    faiss_weight: float = 0.5
    rerank_top_k: int = 5

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.k <= 0:
            raise ValueError("k must be positive")
        if not (0 <= self.bm25_weight <= 1):
            raise ValueError("bm25_weight must be between 0 and 1")
        if not (0 <= self.faiss_weight <= 1):
            raise ValueError("faiss_weight must be between 0 and 1")
        if abs(self.bm25_weight + self.faiss_weight - 1.0) > 1e-6:
            raise ValueError("bm25_weight + faiss_weight must equal 1.0")
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be positive")



@dataclass
class LLMConfig:
    """Configuration for Large Language Model."""
    
    model_type: str
    model_name: str
    model_url: str
    model_api_key: str
    max_tokens: int = 512
    temperature: float = 0.1
    timeout: int = 30
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not all([self.model_type, self.model_name, self.model_url]):
            raise ValueError("model_type, model_name, and model_url are required")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not (0 <= self.temperature <= 2):
            raise ValueError("temperature must be between 0 and 2")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create LLMConfig from environment variables."""
        return cls(
            model_type=os.getenv("LLM_MODEL_TYPE", ""),
            model_name=os.getenv("LLM_MODEL_NAME", ""),
            model_url=os.getenv("LLM_MODEL_URL", "localhost"),
            model_api_key=os.getenv("LLM_MODEL_API_KEY", ""),
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            timeout=int(os.getenv("LLM_TIMEOUT", "30"))
        )


@dataclass
class RecursiveCharacterSplitterConfig:
    """Configuration for recursive character text splitter."""
    type: Literal["recursive_character"] = "recursive_character"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    #     # Chunking
    # if self.chunk_size <= 0:
    #     raise ValueError("chunk_size must be positive")
    # if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
    #     raise ValueError("chunk_overlap must be non-negative and less than chunk_size")



@dataclass
class DocumentsConfig:
    """Configuration for loading and processing document files (e.g., PDFs)."""

    # Path to a single file or directory to scan
    file_path: str

    # Which loader to use: PyMuPDF, pdfminer.six, or Unstructured
    loader_type: str = "PyMuPDF"

    # Which file extensions to ingest
    file_extensions: List[str] = field(default_factory=lambda: [".pdf"])

    # PDF-specific options
    password: Optional[str] = None
    pages: Optional[List[int]] = None  # e.g. [1,2,5] or None for all

    # Any extra metadata fields to attach to each chunk
    metadata: Dict[str, str] = field(default_factory=dict)

    # Timeout (in seconds) for any I/O operations
    timeout: int = 30

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Required
        if not self.file_path:
            raise ValueError("file_path is required")


        # File extensions
        if not self.file_extensions:
            raise ValueError("file_extensions must not be empty")

        # Timeout
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

        # Loader type
        supported = {"PyMuPDF", "pdfminer", "unstructured"}
        if self.loader_type not in supported:
            raise ValueError(
                f"Unsupported loader_type '{self.loader_type}'. "
                f"Choose one of {supported}"
            )

        # Pages list
        if self.pages is not None:
            if not all(isinstance(p, int) and p > 0 for p in self.pages):
                raise ValueError("pages must be a list of positive integers")

    @classmethod
    def from_env(cls) -> "DocumentsConfig":
        """Create DocumentsConfig from environment variables."""
        # Helper to parse comma-separated ints
        def _parse_int_list(env_val: Optional[str]) -> Optional[List[int]]:
            if not env_val:
                return None
            nums = [s.strip() for s in env_val.split(",") if s.strip()]
            return [int(n) for n in nums]

        return cls(
            file_path=os.getenv("DOC_FILE_PATH", ""),
            loader_type=os.getenv("DOC_LOADER_TYPE", "PyMuPDF"),
            file_extensions=os.getenv("DOC_EXTENSIONS", ".pdf").split(","),
            password=os.getenv("DOC_PASSWORD") or None,
            pages=_parse_int_list(os.getenv("DOC_PAGES")),
            timeout=int(os.getenv("DOC_TIMEOUT", "30")),
        )


@dataclass
class SemanticSplitterConfig:
    """Configuration for semantic text splitter."""
    type: Literal["semantic"] = "semantic"
    breakpoint_threshold_type: Literal["percentile", "standard_deviation", "interquartile"] = "percentile"
    breakpoint_threshold_amount: float = 0.95

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not (0 < self.breakpoint_threshold_amount <= 1):
            raise ValueError("breakpoint_threshold_amount must be between 0 and 1")


@dataclass
class RAGConfig:
    """Complete RAG system configurations"""
    documents_config: DocumentsConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    text_splitter: RecursiveCharacterSplitterConfig
    retrieval: RetrievalConfig
    llm: LLMConfig

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create RAGConfig from environment variables with defaults."""
        # 1. Instantiate directly:
        documents_config = DocumentsConfig.from_env()
        # (
        #     file_path="data/contracts/",
        #     loader_type="PyMuPDF",
        #     file_extensions=[".pdf", ".PDF"]
        # )
        return cls(
            documents_config=documents_config,
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            text_splitter=RecursiveCharacterSplitterConfig(),
            retrieval=RetrievalConfig(),
            llm=LLMConfig.from_env()
        )
