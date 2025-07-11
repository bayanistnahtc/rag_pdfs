import logging
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from data_processing.base import BaseLoader
from configs import DocumentsConfig, RecursiveCharacterSplitterConfig, VectorStoreConfig


logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    """
    A downloader for PDF documents using PyMuPDF.
    Provides basic text extraction.
    """

    def __init__(
            self,
            documents_config: DocumentsConfig,
            chunking_config: RecursiveCharacterSplitterConfig,
            vector_store_config: VectorStoreConfig
    ):
        self.documents_config = documents_config
        self.chunking_config = chunking_config
        self.vector_store_config = vector_store_config
        self._text_splitter = None
        self.save_cache = False
    
    @property
    def text_splitter(self):
        """
        Initialization of text splitter
        
        Returns:
            Configured text splitter instance
        """
        if self._text_splitter is None:
            try:
                self._text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunking_config.chunk_size, 
                    chunk_overlap=self.chunking_config.chunk_overlap, 
                    add_start_index=True,
                    length_function=len,
                )
            except Exception as e:
                logger.error(f"Failed to initialize text_splitter: {e}")
                raise
        return self._text_splitter

    def load_documents(self, use_local: bool = False) -> List[Document]:
        """
        Load all PDF documents under a directory (or single file).
        If use_local is True, pull from cache instead.
        """
        try:
            # NOTE: PyMuPDFLoader handles text PDFs well,
            # but scanned documents will require OCR, such as Tesseract.
            # TODO: Add OCR logic in the future, possibly via Unstructured.

            if use_local:
                documents = self._load_local_documents()
            else:
                documents = self._load_pdf_documents()
                if not documents:
                    raise ValueError(f"No PDF files found at {self.documents_config.file_path!r}")
                if self.save_cache:
                    self._cache_documents(documents)
            
            if not documents:
                raise ValueError("No documents were loaded")

            logger.info(f"Successfully loaded {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Failed to load PDF {documents[0]}: {e}")
            # NOTE: For now we're just throwing an exception, but we need more granular error handling.
            raise

    def process_documents(self, documents):
        if not documents:
            raise ValueError("No documents provided for processing")
        
        # TODO: add documents cleaning
        try:
            all_chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")
            
            return all_chunks
        except Exception as e:
            logger.error(f"Failed to peocess documents: {e}")
            raise
    
    def _load_local_documents(self) -> List[Document]:
        """
        Load documents from local cache
        """
        raise NotImplementedError("Local cache loading is not implemented")

    def _load_pdf_documents(self) -> List[Document]:
        """
        Walk self.documents_config.file_path (file or directory),
        load all matching PDFs via PyMuPDF, annotate metadata, and return.
        """
        # try:
        root = Path(self.documents_config.file_path)
        exts = {e.lower() for e in self.documents_config.file_extensions}
        all_docs: List[Document] = []

        if not root.exists():
            raise ValueError(f"Path does not exist: {root!r}")
        
        # Gather all .pdf files under the directory (or single file)
        if root.is_dir():
            pdf_files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
        else:
            pdf_files = [root] if root.suffix.lower() in exts else []

        if not pdf_files:
            return []

        for pdf_path in pdf_files:
            try:
                loader = PyMuPDFLoader(file_path=str(pdf_path))
                docs = loader.load()
                
                
                # Annotate each chunk with source metadata
                for doc in docs:
                    doc.metadata.update({
                        "file_name": pdf_path.name,
                        "file_path": str(pdf_path),
                        "file_type": "pdf",
                    })

                all_docs.extend(docs)
                logger.debug(f"Loaded {len(docs)} pages from {pdf_path.name}")

            except Exception as e:
                logger.error(f"Error loading {pdf_path!r}: {e}")
                # continue with next file rather than failing everything
                continue


        #     # NOTE: For now we're tested the reading only on one pdf file
        #     # TODO: Implement support of reading list of pdf ducuments

        #     loader = PyMuPDFLoader(file_path=self.documents_config.sources[0])
        #     documents = loader.load()
        #     logger.info(f"Loaded {len(documents)} documents from list of PDF files")
        #     return documents
        # except Exception as e:
        #     logger.error(f"Failed to load documend from List of pdfs: {e}")
        #     raise
        return all_docs
    