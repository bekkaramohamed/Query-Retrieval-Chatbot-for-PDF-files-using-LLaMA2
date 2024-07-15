import pickle
from data_ingestion import *
from langchain.prompts import PromptTemplate
DB_FAISS_PATH = "vector_stores/db_faiss"
pickle._allow_dangerous_deserialization = True

# Custom prompt template for QA retrieval
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer and just apologize.

Context: {context}
Question: {question}

Only return the helpful and correct answer below and nothing else.
Helpful and correct answer: 
"""

# set custom prompt
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Load LLM
def load_llm():
    llm = CTransformers(
        model="llama2/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=500,
        temperature=0.5
    )
    return llm

# Set up retrieval QA chain
def retrieval_qa_chain(llm, prompt, db):
    """
    Create and configure a RetrievalQA chain for answering questions.

    Args:
        llm: The language model used for generating answers.
        prompt: The prompt template used for generating answers.
        db: The database from which to retrieve relevant documents.

    Returns:
        qa_chain: A RetrievalQA chain object configured with the given parameters.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # The language model for generating answers
        chain_type='stuff',  # The type of chain, 'stuff' for basic QA
        retriever=db.as_retriever(search_kwargs={'k': 2}),  # Document retriever configured to return top 2 results
        return_source_documents=True,  # Include source documents in the response
        chain_type_kwargs={'prompt': prompt}  # Additional arguments for configuring the chain
    )
    return qa_chain


# Initialize QA bot
def qa_bot():
    """
    Initialize the QA bot by setting up embeddings, loading the database,
    configuring the language model, and setting the QA prompt.

    Returns:
        qa: A RetrievalQA chain object ready for answering queries.
    """
    # Create embeddings using HuggingFace's sentence-transformers model
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',  
        model_kwargs={'device': 'cpu'}  # Use CPU for computation
    )
    
    # Load the FAISS index for document retrieval
    db = FAISS.load_local(
        DB_FAISS_PATH,  # Path to the FAISS index
        embeddings,  # Embeddings for document representation
        allow_dangerous_deserialization=True  # Allow deserialization even if potentially unsafe
    )
    
    # Load the language model for generating answers
    llm = load_llm()
    
    # Set the custom prompt template for the QA chain
    qa_prompt = set_custom_prompt()
    
    # Create and return the QA chain
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


# Get final result
async def final_result(query):
    """
    Get the final result for a given query by using the QA bot.

    Args:
        query: The question or query for which an answer is required.

    Returns:
        response: The answer to the query including source documents.
    """
    # Initialize the QA bot
    qa_result = qa_bot()
    
    # Get the response from the QA bot asynchronously
    response = await qa_result({'query': query})
    
    return response


@cl.on_chat_start
async def start():
    # Initialize the QA bot chain
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    
    msg.content = 'Retrieval bot! Please enter your query.'
    await msg.update()
    
    # Store the chain object in the user session for later use
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    # Retrieve the chain object from the user session
    chain = cl.user_session.get("chain")
    print("The chain is", chain)  # Log the current chain for debugging
    
    # Initialize the callback handler for processing responses
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,  # Enable streaming of the final answer
        answer_prefix_tokens=["FINAL", "Answer"]  # Tokens to identify the final answer
    )
    
    # Set the callback to indicate that the answer has been reached
    cb.answer_reached = True
    
    # Invoke the chain with the user's message content and the callback handler
    res = await chain.ainvoke(message.content, callbacks=[cb])
    
    # Extract the result and source documents from the response
    answer = res["result"]
    sources = res["source_documents"]

    # Format the sources if they are available
    if sources:
        formatted_sources = "\n".join(
            [f"- {source.metadata['source']} (page {source.metadata.get('page', 'N/A')})" for source in sources]
        )
        answer += f"\n\nSources:\n{formatted_sources}"
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()
