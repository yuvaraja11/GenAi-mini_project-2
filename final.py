import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
import psycopg2
from psycopg2.extras import execute_values
import google.generativeai as genai
import PyPDF2
import uuid
from pgvector.psycopg2 import register_vector
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import re
from urllib.robotparser import RobotFileParser


load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")


genai.configure(api_key=GOOGLE_API_KEY)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
gemini_model = genai.GenerativeModel("gemini-1.5-pro")


if 'processed_data' not in st.session_state:
    st.session_state.processed_data = False
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'previous_source' not in st.session_state:
    st.session_state.previous_source = None

def setup_database():
    """
    Connect to PostgreSQL database and set up necessary extensions and tables
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cursor = conn.cursor()
        
       
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        
      
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            chunk_id TEXT,
            document_id TEXT,
            source_type TEXT,
            source_name TEXT,
            chunk_text TEXT,
            embedding vector(384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        return conn, cursor
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None

def clear_database(conn, cursor, document_id=None):
    """
    Clear all data or specific document data from the database
    """
    try:
        if document_id:
            cursor.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
        else:
            cursor.execute("DELETE FROM document_chunks")
        conn.commit()
    except Exception as e:
        st.error(f"Error clearing database: {e}")

def init_selenium():
    """
    Initialize Selenium WebDriver for dynamic content scraping
    """
    try:
        options = Options()
        options.add_argument("--headless")  
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        st.error(f"Failed to initialize Selenium: {e}")
        return None

def is_allowed_by_robots(url):
    """
    Check if scraping is allowed by robots.txt
    """
    try:
        parsed_url = urllib.parse.urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        rp = RobotFileParser()
        rp.set_url(f"{base_url}/robots.txt")
        rp.read()
        
        return rp.can_fetch("*", url)
    except:
        
        return True

def normalize_url(url):
    """
    Normalize URLs to avoid duplicates
    """
    parsed = urllib.parse.urlparse(url)
    
   
    scheme = parsed.scheme.lower()
    
    
    path = re.sub(r'/(index|default)\.(html|htm|php|asp)$', '/', parsed.path)
    
 
    path = path.rstrip('/')
    
    
    fragment = ''
    
   
    normalized = urllib.parse.urlunsplit((
        scheme,
        parsed.netloc.lower(),
        path,
        parsed.query,
        fragment
    ))
    
    return normalized

def scrape_website(base_url, document_id, conn, cursor, max_pages=20):
    """
    Recursively scrape a website by following internal links and store each page's data in the database
    
    Args:
        base_url: The starting URL to scrape
        document_id: Unique ID for the document
        conn: Database connection
        cursor: Database cursor
        max_pages: Maximum number of pages to scrape
        
    Returns:
        Number of pages scraped and stored
    """
    visited = set()
    to_visit = [(base_url, 0)] 
    pages_scraped = 0
    depth_limit = 3  
    
   
    base_domain = urllib.parse.urlparse(base_url).netloc
    
    with st.spinner(f"Processing website data..."):
     
        while to_visit and (max_pages is None or pages_scraped < max_pages):
           
            current_url, current_depth = to_visit.pop(0)
            
          
            if current_url in visited or (depth_limit is not None and current_depth > depth_limit):
                continue
            
           
            if not is_allowed_by_robots(current_url):
                continue
            
          
            driver = init_selenium()
            try:
                driver.get(current_url)
                
             
                delay = 1 + np.random.random() * 2
                time.sleep(delay)
                
          
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')
                
               
                for element in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
                    element.extract()
                
                
                main_content = ""
                
                
                for container in soup.select("main, article, #content, .content, #main, .main"):
                    container_text = container.get_text(separator=' ', strip=True)
                    if len(container_text) > len(main_content):
                        main_content = container_text
                
                
                if not main_content:
                    main_content = soup.get_text(separator=' ', strip=True)
                
               
                main_content = re.sub(r'\s+', ' ', main_content).strip()
                
               
                page_text = f"URL: {current_url}\n{main_content}"
                
               
                if page_text:
                 
                    chunks = chunk_text(page_text, "url", current_url, document_id)
                    
                
                    embedded_chunks = embed_chunks(chunks)
                    store_in_database(conn, cursor, embedded_chunks)
                
               
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    
                  
                    if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                        continue
                    
                
                    full_url = urllib.parse.urljoin(current_url, href)
                    parsed_url = urllib.parse.urlparse(full_url)
                    
                  
                    if parsed_url.netloc == base_domain:
                       
                        normalized_url = normalize_url(full_url)
                        
                        if normalized_url not in visited and not any(url == normalized_url for url, _ in to_visit):
                            to_visit.append((normalized_url, current_depth + 1))
            
            except Exception as e:
                
                pass
            finally:
                driver.quit()
                
            
            visited.add(current_url)
            pages_scraped += 1
            
           
            if pages_scraped >= max_pages:
                break
    
    return pages_scraped

def chunk_text(text, source_type, source_name, document_id):
    """
    Split text into manageable chunks with overlap
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        processed_chunks.append({
            "chunk_id": chunk_id,
            "document_id": document_id,
            "source_type": source_type,
            "source_name": source_name,
            "chunk_text": chunk
        })  
    return processed_chunks

def embed_chunks(chunks):
    """
    Create vector embeddings for text chunks
    """
    texts = [chunk["chunk_text"] for chunk in chunks]
    
    
    batch_size = 64
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embed_model.encode(batch_texts)
        all_embeddings.extend(batch_embeddings)
    
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = all_embeddings[i]
    
    return chunks

def store_in_database(conn, cursor, embedded_chunks):
    """
    Store embedded chunks in PostgreSQL database
    """
    try:
      
        insert_query = """
        INSERT INTO document_chunks (chunk_id, document_id, source_type, source_name, chunk_text, embedding)
        VALUES %s
        """
        values = [(
            chunk["chunk_id"],
            chunk["document_id"],
            chunk["source_type"],
            chunk["source_name"],
            chunk["chunk_text"],
            np.array(chunk["embedding"])
        ) for chunk in embedded_chunks]
        
       
        execute_values(cursor, insert_query, values)
        conn.commit()
    except Exception as e:
        st.error(f"Error storing data in database: {e}")


def process_pdf(pdf_file):
    """Process PDF file and extract text"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text().replace('\x00', '')
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def process_text_file(text_file):
    """Process text file and extract content"""
    try:
        text = text_file.getvalue().decode("utf-8").replace('\x00', '')
        return text
    except Exception as e:
        st.error(f"Error processing text file: {e}")
        return None

def process_csv_file(csv_file):
    """Process CSV file and convert to text representation"""
    try:
        df = pd.read_csv(csv_file)
        
        text = df.to_string().replace('\x00', '')
        return text
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None

def query_documents(conn, cursor, query, top_k=5):
    """
    Query the database for relevant chunks and generate an answer
    """
    try:
      
        query_embedding = embed_model.encode(query)
        query_embedding = np.array(query_embedding).flatten().tolist()

        
        search_query = """
        SELECT chunk_text, source_name, 1 - (embedding <=> %s::vector) AS similarity
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """

        cursor.execute(search_query, (query_embedding, query_embedding, top_k))
        results = cursor.fetchall()

        if not results:
            return "No relevant information found. Please try a different question or add more documents."

     
        context = "\n\n".join([f"From {result[1]}:\n{result[0]}" for result in results])

       
        sources = {}
        for i, result in enumerate(results, 1):
            source = result[1]
            similarity = result[2]
            sources[source] = max(sources.get(source, 0), similarity)

        sorted_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)

        source_references = "\n".join(
            [f"{i+1}. {source} (similarity: {similarity:.4f})"
             for i, (source, similarity) in enumerate(sorted_sources)]
        )

       
        prompt = f"""
        Based on the following context, provide a comprehensive answer to the question.
        Give answer in more lines and provide more detailed information.
        Reference specific sources when providing information.

        Context:
        {context}

        Question: {query}

        Answer:
        """

        response = gemini_model.generate_content(prompt)

      
        final_response = f"{getattr(response, 'text', response)}\n\n**Sources:**\n{source_references}"
        return final_response

    except Exception as e:
        st.error(f"Error querying documents: {e}")
        return f"An error occurred: {str(e)}"


def check_source_change(source_type, source_name):
    """
    Check if data source has changed and clear conversation history if needed
    """
    current_source = f"{source_type}:{source_name}"
    
    if st.session_state.previous_source is not None and current_source != st.session_state.previous_source:
        st.session_state.qa_history = []  
        
    st.session_state.previous_source = current_source


def main():
    
    st.set_page_config(layout="wide", page_title="RAG GenAI QA System")
    
   
    with st.sidebar:
        st.header("previous History")
        st.write("Previous questions and answers will appear here.")
        
        if not st.session_state.qa_history:
            st.info("No conversation history yet.")
        else:
            for i, (q, a) in enumerate(st.session_state.qa_history):
                with st.expander(f"Q: {q[:50]}{'...' if len(q) > 50 else ''}", expanded=False):
                    st.markdown(f"**Question:**\n{q}")
                    st.markdown(f"**Answer:**\n{a}")
    
    
    st.title("RAG model & GenAI Question Answering System")
    st.write("Upload a file or provide a URL to extract information, then ask questions about the content.")
    
    
    conn, cursor = setup_database()
    if not conn or not cursor:
        st.error("Failed to connect to database. Please check your database configuration.")
        return
    
 
    st.subheader("Step 1: Choose Your input Source")
    
    col1, col2 = st.columns(2)
    
    with col1:
        url_button = st.button("Use URL", use_container_width=True)
    
    with col2:
        file_button = st.button("Upload File", use_container_width=True)
    
    
    if url_button:
        st.session_state.data_source = "url"
        if st.session_state.previous_source is not None and st.session_state.previous_source != "url":
          
            clear_database(conn, cursor)
            st.session_state.qa_history = [] 
    elif file_button:
        st.session_state.data_source = "file"
        if st.session_state.previous_source is not None and st.session_state.previous_source != "file":
          
            clear_database(conn, cursor)
            st.session_state.qa_history = [] 
    
   
    if st.session_state.data_source == "url":
        st.subheader("Enter URL to Scrape")
        url = st.text_input("Website URL")
        
        max_pages = 20
        
        if url and st.button("Process URL"):
        
            clear_database(conn, cursor)
            
            
            st.session_state.qa_history = []
            
        
            document_id = str(uuid.uuid4())
            
            with st.spinner("Processing website data. This may take a few minutes..."):
             
                pages_scraped = scrape_website(url, document_id, conn, cursor, max_pages)
                
                if pages_scraped > 0:
                    st.success(f"Successfully processed website data")
                    st.session_state.processed_data = True
                    check_source_change("url", url)
    
  
    elif st.session_state.data_source == "file":
        st.subheader("Upload a File")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv"])
        
        if uploaded_file and st.button("Process File"):
           
            clear_database(conn, cursor)
            
           
            st.session_state.qa_history = []
            
            
            document_id = str(uuid.uuid4())
           
            file_type = uploaded_file.name.split(".")[-1].lower()
            if file_type == "pdf":
                text_content = process_pdf(uploaded_file)
                source_name = uploaded_file.name
            elif file_type == "txt":
                text_content = process_text_file(uploaded_file)
                source_name = uploaded_file.name
            elif file_type == "csv":
                text_content = process_csv_file(uploaded_file)
                source_name = uploaded_file.name
            else:
                st.error("Unsupported file type")
                text_content = None
                source_name = None
            
            if text_content:
                with st.spinner("Processing file data..."):
               
                    chunks = chunk_text(text_content, "file", source_name, document_id)
                    embedded_chunks = embed_chunks(chunks)
                    store_in_database(conn, cursor, embedded_chunks)
                    st.session_state.processed_data = True
                    check_source_change("file", source_name)
                st.success(f"Successfully processed {uploaded_file.name}")
    
    
    if st.session_state.processed_data:
        st.subheader("Ask Questions About Your Data")
        query = st.text_input("Enter your question:")
        
     
        num_results = 5
        
        if query and st.button("Get Answer"):
            with st.spinner("Generating answer..."):
                answer = query_documents(conn, cursor, query, top_k=num_results)
                
               
                st.session_state.qa_history.append((query, answer))
                
                st.subheader("Answer:")
                st.markdown(answer)
    
   
    st.markdown("---")
    st.caption("RAG GenAI Question Answering System powered by Gemini and pgvector")
    
    
    if conn:
        conn.close()

if __name__ == "__main__":
    main()