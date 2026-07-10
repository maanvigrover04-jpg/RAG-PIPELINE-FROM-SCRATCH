import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
import os 
import uuid 
from datetime import datetime
import json

load_dotenv()      #loads the open ai key from .env to environment variables
client_openai = OpenAI(
    base_url="https://openrouter.ai/api/v1",   
    api_key=os.getenv("OPENROUTER_API_KEY")
    )                                              

# Connect to the SAME persistent client and collection used in ingestion
client = chromadb.PersistentClient(path="chromadb")                                # to match ingestion's path exactly

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"                                                  # to match ingestion's model exactly
)

collection = client.get_or_create_collection(
    name="documents_collection",                                                    # to match ingestion's collection name exactly
    embedding_function=sentence_transformer_ef
)


def semantic_search(collection , query: str, n_results: int=3):
    """perform semantic search on collection """
    results = collection.query(
        query_texts = [query],
        n_results=n_results
    )
    return results

def get_context_with_sources(results):
    """extract content and source from results"""
    #combine document chunks into simple context
    context = "\n\n".join(results['documents'][0])
    
    #format sources
    sources = [
        f"{meta['source']} (chunk {meta['chunk']})" 
        for meta in results['metadatas'][0]
    ]

    return context, sources

def get_prompt(context: str, conversation_history: str, query: str):
    """generate a prompt based on context , history and query"""
    prompt = f"""Based on the following context and conversation history, 
    please provide a relevant and contextual response. If the answer cannot 
    be derived from the context, only use the conversation history or say 
    "I cannot answer this based on the provided information."


    Context from documents:
    {context}

    Previous conversation:
    {conversation_history}

    Human: {query}

    Assistant:"""
    
    return prompt


def generate_response(query: str,context: str, conversation_history: str=" "):
    """generate a response using open ai using concersation history"""
    prompt = get_prompt(context, conversation_history, query)    
    try:
        response = client_openai.chat.completions.create(           
            model="openrouter/free",      
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."}, #seperates behavioural instruction from actual answer
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # temperature controls randomness/creativity in the output. 0 makes the model as deterministic and focused as possible — same question in, (nearly) same answer out every time. For a document Q&A system, you want factual, consistent answers based on the retrieved context, not creative variation
            max_tokens=500   #keeps answers concice
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"
    
    
def rag_query(collection, query: str, conversation_history: str = "", n_chunks: int = 3):
    results = semantic_search(collection, query, n_chunks)
    context, sources = get_context_with_sources(results)
    response = generate_response(query, context, conversation_history)
    updated_history = conversation_history + f"\nHuman: {query}\nAssistant: {response}"
    return response, sources, updated_history

            
            
#session management 

conversations = {}
def create_session():
    """ create a new conversation session"""
    session_id = str(uuid.uuid4())
    conversations[session_id] = []
    return session_id


def add_message(session_id: str, role: str, content: str):
    """Add a message to the conversation history"""
    if session_id not in conversations:
        conversations[session_id] = []

    conversations[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

def get_conversation_history(session_id: str, max_messages: int = None):
    """Get conversation history for a session"""
    if session_id not in conversations:
        return []

    history = conversations[session_id]
    if max_messages:
        history = history[-max_messages:]

    return history


#create a formatter for conversation history 
def format_history_for_prompt(session_id: str , max_messages: int = 5):
    """formats conversation history for inclusion in prompts"""
    history = get_conversation_history(session_id, max_messages)
    formatted_history = " "
    for msg in history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n\n"
        
    return formatted_history.strip()


def contextualize_query(query: str, conversation_history: str, client: OpenAI):
    """Convert follow-up questions into standalone queries"""
    contextualize_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone 
    question which can be understood without the chat history. Do NOT answer 
    the question, just reformulate it if needed and otherwise return it as is."""   #This is the system instruction — telling the model exactly one job: rewrite the question so it stands alone, don't actually answer it. The "otherwise return it as is" part matters — if the question is already standalone

    try:
        completion = client.chat.completions.create(          #making an LLM call
            model = "openrouter/free", 
            messages = [
                {"role": "system", "content": contextualize_prompt},
                {"role": "user", "content": f"Chat history:\n{conversation_history}\n\nQuestion:\n{query}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error contextualizing query:  {str(e)}" )
        return query  #fallback to original query 
    
    
def get_prompt(context, conversation_history, query):
    prompt = f"""Based on the following context and conversation history, please provide a relevant and contextual response.
    If the answer cannot be derived from the context, only use the conversation history or say "I cannot answer this based on the provided information."

    Context from documents:
    {context}

    Previous conversation:
    {conversation_history}

    Human: {query}

    Assistant:"""
    return prompt




def conversational_rag_query(
    collection,
    query: str,
    session_id: str,
    n_chunks: int = 3
):
    """Perform RAG query with conversation history"""
    # Get conversation history
    conversation_history = format_history_for_prompt(session_id)
    
    # Handle follow up questions
    query = contextualize_query(query,conversation_history,client_openai)
    print("Contextualized Query:", query)
    
    # Get relevant chunks
    context, sources = get_context_with_sources(
        semantic_search(collection, query, n_chunks)
    )
    print("Context:", context)
    print("Sources:", sources)

    response = generate_response(query, context, conversation_history)
    
    # Add to conversation history
    add_message(session_id, "user", query)
    add_message(session_id, "assistant", response)

    return response, sources


if __name__ == "__main__":
    session_id = create_session()
    
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        response, sources = conversational_rag_query(collection, query, session_id)
        
        print("\nAnswer:", response)
        print("Sources used:")
        for source in sources:
            print(f"- {source}")




      
        
