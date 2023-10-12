from langchain import hub

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings import OllamaEmbeddings  # We can also try Ollama embeddings
from langchain.llms import Ollama
from langchain.schema import LLMResult
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


CONFIG = {
    "llama2-7b": {"model_name": "llama2", "prompt_method": "rlm/rag-prompt-llama"},
    "mistral-7b": {
        "model_name": "mistral:7b-instruct",
        "prompt_method": "rlm/rag-prompt-mistral",
    },
}

MODEL = "llama2-7b"
# MODEL = "mistral-7b"
_config = CONFIG[MODEL]


class GenerationStatisticsCallback(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        gen_info = response.generations[0][0].generation_info

        print(
            "\n\nSTATISTICS:\n" + "=" * len("statistics:"),
            "Tok / s: "
            + str(gen_info["eval_count"] / (gen_info["eval_duration"] / 1e9)),
            sep="\n",
        )


callback_manager = CallbackManager(
    [StreamingStdOutCallbackHandler(), GenerationStatisticsCallback()]
)

# Load web page
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

# Embed and store
# do try sentence_embedder or sth
vectorstore = Chroma.from_documents(
    documents=all_splits,
    # Faster, lower quality?
    embedding=GPT4AllEmbeddings()  # gpt4all/ggml-all-MiniLM-L6-v2-f16
    # Slower, higher quality?
    # embedding=OllamaEmbeddings() # llama2
)

# Retrieve
question = "How can Task Decomposition be done?"
docs = vectorstore.similarity_search(question)
print(len(docs))


# RAG prompt
QA_CHAIN_PROMPT = hub.pull(_config["prompt_method"])


# LLM
llm = Ollama(
    model=_config["model_name"],
    # verbose=True,
    callback_manager=callback_manager,
)

# llm("Tell me about the history of AI.")


# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

questions = [
    "What are the various approaches to Task Decomposition for AI Agents?",
    "What is this document about?",
]

for question in questions:
    print("Question:", question)
    result = qa_chain({"query": question})
