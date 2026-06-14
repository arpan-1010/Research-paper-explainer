from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import (ChatHuggingFace, HuggingFaceEndpoint)

from langchain_core.output_parsers import StrOutputParser

from prompts import paper_prompt

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",
    temperature=0.3,
    max_new_tokens=1024
)
chat_model = ChatHuggingFace(llm=llm)

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

def create_chain(vector_db):

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 8}
    )

    chain = (
        {
            "context": retriever | format_docs,
        }
        | paper_prompt
        | chat_model
        | StrOutputParser()
    )

    return chain