"""
Interactive RAG for the command line.
"""
import argparse
import os
from typing import Any, Dict, List, Union
from uuid import UUID

from langchain import hub
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings import GPT4AllEmbeddings
# from langchain.embeddings import OllamaEmbeddings  # We can also try Ollama embeddings
from langchain.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import LLMResult
from langchain.schema.messages import BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

MODEL_NAME = "mistral:7b-instruct"
PROMPT_METHOD = "rlm/rag-prompt-mistral"

INITIAL_PROMPT = r"""
You are an experienced researcher, expert at interpreting and answering questions based on provided sources. Using the provided context, answer the user's question to the best of your ability using the resources provided.
Generate a concise answer for a given question based solely on the provided search results (URL and content). You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. Do not repeat text.
If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." Don't try to make up an answer.
Anything between the following \`context\` html blocks is retrieved from a knowledge bank, not part of the conversation with the user.
<context>
    {context}
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm not sure." Don't try to make up an answer. Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user.
""".strip()

HISTORY_PROMPT = r"""
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:
""".strip()

initial_prompt = PromptTemplate.from_template(INITIAL_PROMPT)
history_prompt = PromptTemplate.from_template(HISTORY_PROMPT)


class InvalidFileArgumentException(Exception):
    """
    The exception raised when an invalid file is parsed
    """

    def __init__(self, message: str):
        self.__cause__ = message


class GenerationStatisticsCallback(BaseCallbackHandler):
    """
    The callback for statistics reporting on the inference
    """

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        gen_info = response.generations[0][0].generation_info

        print(
            "\n\nSTATISTICS:\n" + "=" * len("statistics:"),
            "Tok / s: "
            + str(gen_info["eval_count"] / (gen_info["eval_duration"] / 1e9)),
            sep="\n",
        )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Abstract implementation
        """
        return super().on_chat_model_start(
            serialized,
            messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )


callback_manager = CallbackManager(
    [StreamingStdOutCallbackHandler(), GenerationStatisticsCallback()]
)


def get_embeddings(file_name: str):
    """
    Processes the input document, returning an in-memory vector store.
    """
    loader = PDFPlumberLoader(file_name)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    vector_store = Chroma.from_documents(
        documents=all_splits, embedding=GPT4AllEmbeddings()
    )
    return vector_store


def get_llm():
    """
    Retrieves the LLM instance from the local Ollama store.
    """
    _llm = Ollama(model=MODEL_NAME, callback_manager=callback_manager)
    return _llm


def get_qa_chain(_llm: Ollama, _embeddings: Chroma):
    """
    Retrieves the main component for the qa process.
    """
    _qa_chain = RetrievalQA.from_chain_type(
        _llm,
        retriever=_embeddings.as_retriever(),
        chain_type_kwargs={"prompt": hub.pull(PROMPT_METHOD)},
    )
    return _qa_chain


def validate_filename(file_name: str):
    """
    Validates the supplied file_path.
    """
    if not file_name or not isinstance(file_name, str):
        raise InvalidFileArgumentException("Invalid argument supplied.")
    if not os.path.exists(file_name):
        raise InvalidFileArgumentException("File not found")
    if not os.access(os.path.abspath(file_name), os.R_OK):
        raise InvalidFileArgumentException("Insufficient permissions to read from file")


def setup() -> str:
    """
    Sets up the arguments for the program, and
    parses the necessary arguments.

    Flags include:
    `--file` or `-f` - The file to input from.

    Returns
    =======
    `file_name`: str
        - The name of the file to ingest data from.
    """
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("--file", "-f", required=True)

    args = parser.parse_args()
    file_name: str = args.file
    validate_filename(file_name)
    print(f"File name: `{file_name}`")
    return file_name


if __name__ == "__main__":
    input_document = setup()
    embeddings = get_embeddings(input_document)
    llm = get_llm()
    qa_chain = get_qa_chain(llm, embeddings)

    while True:
        print("\n================")
        print("Enter Question: ")
        print(">>>", end=" ")
        try:
            question = input()
            if not question:
                break
            result = qa_chain({"query": question})
        except (EOFError, KeyboardInterrupt):
            break

    print("\nExiting. Have a nice day!")
