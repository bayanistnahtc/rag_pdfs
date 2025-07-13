from typing import Any, Dict, List

from langchain_core.documents import Document

from rag.models import Answer


def format_documents(docs: List[Document]) -> str:
    """
    Format retrieved documents for context injection.

    Args:
        docs (List[Document]): list of Document instances retrieved by the vector store.

    Returns: str
        A formatted string with each document chunk prefixed by its source tag.
    """
    
    if not docs:
        return "No relevant context found."
    
    formatted: List[str] = []
    for i, doc in enumerate(docs):
        doc_id = doc.metadata.get('source', 'UNKNOWN')
        page_num = doc.metadata.get('page_label', 'UNKNOWN')
        source_tag = f"[{doc_id}/PAGE_{page_num}]"
        formatted.append(f"Source {i+1} {source_tag}:\n{doc.page_content}")
    return "\n\n".join(formatted)


def docs_to_answer(data: Dict[str, Any]) -> Answer:
    """
    Transform chain output to Answer model.

    Args:
        data (Dict[str, Any]): The dictionary expected to have keys 'answer_text' and 'retrieved_docs'.

    Returns: Answer
        An Answer instance encapsulating the LLM response and its sources.
    """
    return Answer(
        answer_text=data["answer_text"] if data["answer_text"] else "No answer",
        source_chunks=data["retrieved_docs"]
    )
