import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document

from configs import DocumentsConfig, RecursiveCharacterSplitterConfig, VectorStoreConfig
from data_processing.base import BaseLoader
from data_processing.loader import PDFLoader #, DocxLoader


logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """
    Factory for creating document loaders by file type.
    """
    _loaders = {
        ".pdf": PDFLoader,
        # TODO: Add support for .docx, .txt and possibly URLs.
        # ".docx": DocxLoader,
    }

    @staticmethod
    def create_documents_loader(
            documents_config: DocumentsConfig,
            chunking_config: RecursiveCharacterSplitterConfig,
            vector_store_config: VectorStoreConfig
    ):
        return PDFLoader(
            documents_config=documents_config,
            vector_store_config=vector_store_config,
            chunking_config=chunking_config
        )

    @classmethod
    def create_loader(cls, source: Path) -> BaseLoader:
        """
        Creates a suitable loader for the file type.
        """
        suffix = source.suffix.lower()
        loader_class = cls._loaders.get(suffix)
        if not loader_class:
            # HACK: Temporary solution. Ideally, a more flexible system of
            # bootloader registration is needed.
            raise ValueError(f"Unsupported file type: {suffix}")
        return loader_class()

    @classmethod
    def load_documents(cls, sources: List[str]) -> List[Document]:
        """
        Loads documents from a list of sources using the appropriate loaders.
        """
        all_docs = []
        for source in sources:
            file_path = Path(source)
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            try:
                loader = cls.create_loader(file_path)
                documents = loader.load(file_path)
                all_docs.extend(documents)
                logger.info(f"Loaded {len(documents)} documents from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load document from {source}: {e}")
                continue
        return all_docs
    