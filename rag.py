from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Config
CHROMA_DIR = "chroma_db"
MODEL = "gpt-4o-mini"

def ask(question):
    """Ask a question and get an answer"""
    
    # Load vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=CHROMA_DIR,
                         embedding_function=embeddings)
    
    # Retrieve relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) #top 5 most similar chunks to the question
    docs = retriever.invoke(question)
    
    # Combine chunks into context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the following context:
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """)
    
    # Get answer from LLM
    llm = ChatOpenAI(model=MODEL, temperature=0)
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response.content}")
    print(f"\nSources: {len(docs)} chunks retrieved")
    
    return response.content

if __name__ == "__main__":
    # some questions about pcos
    ask("What are the diagnostic criteria for PCOS?")
    print("\n" + "="*50 + "\n")
    ask("How does PCOS affect insulin resistance?")
    print("\n" + "="*50 + "\n")
    ask("What are treatment options for PCOS?")