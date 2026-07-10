import docx
import PyPDF2
import os
import chromadb
from chromadb.utils import embedding_functions     #chromadb is main cient library, emedding_functions is a submodule 
import pandas as pd 

#Ingestion pipeline part 1 - Document processing and indexing

def read_csv_file(file_path: str, text_column: str = None):
    df = pd.read_csv(file_path)
    
    if text_column is None:
        text_column = guess_text_column(df)
        print(f"Auto-detected text column for {file_path}: '{text_column}'")
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available columns: {list(df.columns)}")
    
    rows_text = df[text_column].dropna().astype(str).tolist()
    return "\n\n".join(rows_text)
    
    
def guess_text_column(df):
    common_names = ["text", "content", "article", "description", "body", "review", "summary"]
    for name in common_names:
        if name in df.columns:
            return name
    # fallback to Option 1's logic if none match
    return max(df.select_dtypes(include='object').columns, 
               key=lambda col: df[col].dropna().astype(str).apply(len).mean())

def read_text_file(file_path: str):
    """ Reads content of a text file"""
    with open(file_path, 'r', encoding = 'utf-8') as file:
        return file.read()
    
def read_pdf_file(file_path: str):
    """ Reads content of any pdf file"""
    text = ""                                 #because pdf files are not structured, we need to read them page by page
    with open(file_path, 'rb') as file:       #pdf is binary so rb
        pdf_reader = PyPDF2.PdfReader(file)   #now pdf reader know how many pages there are or whats the structure
        for page in pdf_reader.pages:
            text+= page.extract_text() +"\n"
    return text 

def read_docx_file(file_path: str):
    """ Reads contents of a word document"""
    doc = docx.Document(file_path)             #doc represents whole document
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])   #short way to write for loop for taking out paragprah names


#we then create a unified function for docment reading 

def read_document(file_path: str, text_column: str = None):
    """ Read the content based on file extension"""
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension == '.txt': 
        return read_text_file(file_path)
    elif file_extension == '.pdf':
        return read_pdf_file(file_path)
    elif file_extension == '.docx':
        return read_docx_file(file_path)
    elif file_extension == '.csv':
        return read_csv_file(file_path, text_column=text_column)   
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
  
  
#once we have raw text we perform chunking
#so we can get more precise semantic search results
#we stay in our LLM's context window 


def split_text(text: str, chunk_size:  int = 500):
    """ Split text into chunks while preserving sentence """
    sentences = text.replace('\n',' ').split('. ')            #replaces blank line with space
    chunks = []
    current_chunk = [] 
    current_size = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        #ensure a proper sentence ending
        if not sentence.endswith('.'):
            sentence += '.'
            
        sentence_size = len(sentence)
        
        #check if adding a sentence would increase chunk size
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    #add the last chunk if it exists 
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks 


#initialize chromadb client 
client = chromadb.PersistentClient(path = "chromadb")   #persistent is a keyword , else we have a EphemeraClient option too , it keep severything in memoery and wipes it as soon as script ends, creates a chroma_db folder automatically and keeps it even if script dosent run

#configuring sentence transformer embeddings
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = "all-MiniLM-L6-v2")    #small fast 384 dimensional vector model

#create or get existing collection 
collection = client.get_or_create_collection(       #get_or_create_collection handles initial creation and subsequent access
    name = "documents_collection",
    embedding_function = sentence_transformer_ef
)

#inserting data in chromadb

def process_document(file_path: str, text_column: str = None):
    """Process a single doc and prepare it for chromadb"""
    try:
        content = read_document(file_path, text_column=text_column)
        chunks = split_text(content)
        file_name = os.path.basename(file_path)
        metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
        return ids, chunks, metadatas 
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []
    
def add_to_collection(collection , ids , texts , metadatas):
    """Add documents to collection in batches"""
    if not texts:
        return
    
    batch_size = 100
    for i in range(0 , len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        collection.add(
            documents = texts[i:end_idx],
            metadatas = metadatas[i:end_idx],
            ids = ids[i:end_idx]
        )
        
def process_and_add_documents(collection , folder_path: str):
    """Process all documents in a folder and add to collection"""
    files = [os.path.join(folder_path, file)
             for file in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, file))]
    for file_path in files:
        print(f"Processing {os.path.basename(file_path)}. . .")
        ids, texts , metadatas = process_document(file_path)
        add_to_collection(collection, ids, texts, metadatas)
        print(f"Added {len(texts)} chunks to collection")
        
        
#initializing chromadb collection 
    
collection = client.get_or_create_collection(
    name = "documents_collection",
    embedding_function = sentence_transformer_ef
)

#process and add documents from folder
folder_path ="data"
process_and_add_documents(collection, folder_path)
    
    
