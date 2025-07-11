import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from .base import BaseTextSplitter
from configs import RecursiveCharacterSplitterConfig


logger = logging.getLogger(__name__)


class RecursiveCharacterTextSplitterWrapper(BaseTextSplitter):
    """
    Wrapper for RecursiveCharacterTextSplitter from LangChain.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        logger.info(f"Splitting {len(documents)} documents...")
        # TODO: Add error handling if documents are empty or invalid.
        return self._splitter.split_documents(documents)

class TextSplitterFactory:
    """
    Factory for creating text splitter instances.
    """
    @staticmethod
    def create(config: RecursiveCharacterSplitterConfig) -> BaseTextSplitter:
        # NOTE: Currently only one splitter type is supported.
        # Can easily be extended to support semantic
        # or other splitter types in the future.
        if config.type == "recursive_character":
            return RecursiveCharacterTextSplitterWrapper(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
        raise ValueError(f"Unknown splitter type: {config.type}")
        