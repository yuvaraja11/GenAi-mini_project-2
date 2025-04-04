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
import random
from datetime import datetime


# Load environment variables
load_dotenv()

# Configuration constants
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# Initialize AI models
genai.configure(api_key=GOOGLE_API_KEY)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# Set up session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = set()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'waiting_for_query' not in st.session_state:
    st.session_state.waiting_for_query = False
if 'document_ids' not in st.session_state:
    st.session_state.document_ids = set()


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
        
        # Enable vector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        
        # Create document chunks table if it doesn't exist
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
    Initialize Selenium WebDriver for dynamic content scraping with rotating user agents
    """
    try:
        # List of user agents to rotate through
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
        ]
        
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument(f"--user-agent={random.choice(user_agents)}")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        st.error(f"Failed to initialize browser: {e}")
        return None


def normalize_url(url):
    """
    Normalize URLs to avoid duplicates
    """
    parsed = urllib.parse.urlparse(url)
    
    # Normalize scheme
    scheme = parsed.scheme.lower()
    
    # Remove index files
    path = re.sub(r'/(index|default)\.(html|htm|php|asp)$', '/', parsed.path)
    
    # Remove trailing slashes
    path = path.rstrip('/')
    
    # Remove fragments
    fragment = ''
    
    # Reconstruct the URL
    normalized = urllib.parse.urlunsplit((
        scheme,
        parsed.netloc.lower(),
        path,
        parsed.query,
        fragment
    ))
    
    return normalized


def scrape_website(base_url, document_id, conn, cursor, max_pages=10):
    """
    Recursively scrape a website by following internal links and store all page data
    
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
    to_visit = [(base_url, 0)]  # (url, depth)
    pages_scraped = 0
    depth_limit = 2  # Limit recursion depth
    
    # Extract base domain
    base_domain = urllib.parse.urlparse(base_url).netloc
    
    with st.spinner(f"Scraping website {base_url}"):
        # Create progress bar
        progress_bar = st.progress(0)
        
        while to_visit and pages_scraped < max_pages:
            # Update progress
            progress_percentage = min(pages_scraped / max_pages, 0.99)
            progress_bar.progress(progress_percentage)
            
            # Get next URL from queue
            current_url, current_depth = to_visit.pop(0)
            
            if current_url in visited or current_depth > depth_limit:
                continue
            
            # Initialize Selenium
            driver = init_selenium()
            if not driver:
                continue
                
            try:
                driver.get(current_url)
                
                # Random delay to avoid detection
                delay = 1 + random.random() * 2
                time.sleep(delay)
                
                # Get page content
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')
                
                # Remove script and style elements
                for element in soup(["script", "style", "meta", "noscript"]):
                    element.extract()
                
                # Extract main content
                main_content = ""
                
                # Try different CSS selectors for main content
                for selector in ["main", "article", "#content", ".content", "#main", ".main", "body"]:
                    containers = soup.select(selector)
                    for container in containers:
                        container_text = container.get_text(separator=' ', strip=True)
                        if len(container_text) > len(main_content):
                            main_content = container_text
                
                # Fallback to whole page if no main content found
                if not main_content:
                    main_content = soup.get_text(separator=' ', strip=True)
                
                # Clean up whitespace
                main_content = re.sub(r'\s+', ' ', main_content).strip()
                
                # Create text with metadata
                page_text = f"URL: {current_url}\n\n{main_content}"
                
                if page_text:
                    # Extract title
                    try:
                        title = soup.title.string if soup.title else current_url
                    except:
                        title = current_url
                    
                    page_text = f"Title: {title}\nURL: {current_url}\n\n{main_content}"
                    
                    # Process and store text
                    chunks = chunk_text(page_text, "url", current_url, document_id)
                    embedded_chunks = embed_chunks(chunks)
                    store_in_database(conn, cursor, embedded_chunks)
                
                # Find links to other pages
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    
                    # Skip non-HTTP links
                    if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                        continue
                    
                    # Convert to absolute URL
                    full_url = urllib.parse.urljoin(current_url, href)
                    parsed_url = urllib.parse.urlparse(full_url)
                    
                    # Only follow internal links (same domain)
                    if parsed_url.netloc == base_domain:
                        normalized_url = normalize_url(full_url)
                        
                        if normalized_url not in visited and not any(url == normalized_url for url, _ in to_visit):
                            to_visit.append((normalized_url, current_depth + 1))
            
            except Exception as e:
                # Silently continue if there's an error with a page
                pass
            finally:
                driver.quit()
                
            visited.add(current_url)
            pages_scraped += 1
            
            # Check if we've reached the limit
            if pages_scraped >= max_pages:
                break
        
        # Complete the progress
        progress_bar.progress(1.0)
    
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
    
    # Process in batches to avoid memory issues
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
        # SQL insert query
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
        
        # Execute batch insert
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
        # Create embedding for the query
        query_embedding = embed_model.encode(query)
        query_embedding = np.array(query_embedding).flatten().tolist()

        # Search for similar chunks
        search_query = """
        SELECT chunk_text, source_name, 1 - (embedding <=> %s::vector) AS similarity
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """

        cursor.execute(search_query, (query_embedding, query_embedding, top_k))
        results = cursor.fetchall()

        if not results:
            return "No relevant information found in the uploaded documents or websites. Please try a different question or add more content."

        # Format context from retrieved chunks
        context = "\n\n".join([f"From {result[1]}:\n{result[0]}" for result in results])

        # Track sources for citation
        sources = {}
        for result in results:
            source = result[1]
            similarity = result[2]
            sources[source] = max(sources.get(source, 0), similarity)

        sorted_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)

        source_references = "\n".join(
            [f"{i+1}. {source}"
             for i, (source, _) in enumerate(sorted_sources)]
        )

        # Create prompt for AI model
        prompt = f"""
        Based on the following context, provide a comprehensive answer to the question.
        Give answer in multiple paragraphs and provide detailed information.
        Reference specific sources when providing information.

        Context:
        {context}

        Question: {query}

        Answer:
        """

        response = gemini_model.generate_content(prompt)

        # Format final response with sources
        final_response = f"{getattr(response, 'text', response)}\n\n**Sources:**\n{source_references}"
        return final_response

    except Exception as e:
        st.error(f"Error querying documents: {e}")
        return f"An error occurred: {str(e)}"


def process_multiple_files(files, conn, cursor):
    """
    Process multiple files and store their content in the database
    """
    processed_count = 0
    status_placeholder = st.empty()
    
    for file in files:
        if file.name in st.session_state.processed_files:
            continue
            
        document_id = str(uuid.uuid4())
        st.session_state.document_ids.add(document_id)
        
        file_type = file.name.split(".")[-1].lower()
        status_placeholder.text(f"Processing {file.name}...")
        
        if file_type == "pdf":
            text_content = process_pdf(file)
        elif file_type == "txt":
            text_content = process_text_file(file)
        elif file_type == "csv":
            text_content = process_csv_file(file)
        else:
            st.warning(f"Unsupported file type: {file_type}")
            continue
            
        if text_content:
            chunks = chunk_text(text_content, "file", file.name, document_id)
            embedded_chunks = embed_chunks(chunks)
            store_in_database(conn, cursor, embedded_chunks)
            st.session_state.processed_files.add(file.name)
            processed_count += 1
    
    status_placeholder.empty()
    return processed_count


def process_multiple_urls(urls, conn, cursor):
    """
    Process multiple URLs and store their content in the database
    """
    processed_count = 0
    for url in urls:
        url = url.strip()
        if not url or url in st.session_state.processed_urls:
            continue
            
        document_id = str(uuid.uuid4())
        st.session_state.document_ids.add(document_id)
        
        pages_scraped = scrape_website(url, document_id, conn, cursor)
        if pages_scraped > 0:
            st.session_state.processed_urls.add(url)
            processed_count += 1
    
    return processed_count


def display_chat_message(message, is_user=False):
    """
    Display a chat message in the UI
    """
    if is_user:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
            <div style="background-color: #F0F2F6; padding: 10px 15px; border-radius: 15px; max-width: 80%;">
                <p style="margin: 0;">{message}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; margin-bottom: 10px;">
            <div style="background-color: #E2F0FE; padding: 10px 15px; border-radius: 15px; max-width: 80%;">
                <p style="margin: 0;">{message}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for chat-like interface
    st.markdown("""
    <style>
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .stButton > button {
        border-radius: 20px;
    }
    .css-1v3fvcr {
        padding-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize database connection
    conn, cursor = setup_database()
    if not conn or not cursor:
        st.error("Unable to initialize the system. Please contact support.")
        return

    # Sidebar for file/URL uploads and settings
    with st.sidebar:
        st.title("📚 Document Assistant")
        st.subheader("Upload Your Knowledge Base")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload files (PDF, TXT, CSV)",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True
        )
        
        # URL input section
        url_input = st.text_area(
            "Enter URLs (one per line)",
            height=100
        )
        
        # Process button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Process Data", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    # Process files
                    files_processed = 0
                    if uploaded_files:
                        files_processed = process_multiple_files(uploaded_files, conn, cursor)
                    
                    # Process URLs
                    urls_processed = 0
                    if url_input:
                        urls = url_input.split('\n')
                        urls_processed = process_multiple_urls(urls, conn, cursor)
                    
                    if files_processed > 0 or urls_processed > 0:
                        st.success(f"Processed {files_processed} files and {urls_processed} URLs")
                        st.session_state.data_loaded = True
                    else:
                        st.warning("No new content processed")
        
        with col2:
            if st.button("Clear All Data", type="secondary", use_container_width=True):
                if conn and cursor:
                    clear_database(conn, cursor)
                    st.session_state.processed_files = set()
                    st.session_state.processed_urls = set()
                    st.session_state.document_ids = set()
                    st.session_state.chat_history = []
                    st.session_state.data_loaded = False
                    st.success("All data cleared")
        
        # Display knowledge base summary
        st.subheader("Knowledge Base")
        if st.session_state.processed_files or st.session_state.processed_urls:
            st.write("📄 Files:")
            for file in st.session_state.processed_files:
                st.write(f"  - {file}")
            
            st.write("🔗 URLs:")
            for url in st.session_state.processed_urls:
                st.write(f"  - {url}")
        else:
            st.info("No data loaded yet. Please upload files or enter URLs.")

    # Main chat interface
    st.title("AI Document Assistant")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align: center; padding: 30px 0;">
                <h3>Welcome to the AI Document Assistant!</h3>
                <p>Upload documents or enter URLs in the sidebar, then ask questions about your content.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.chat_history:
                display_chat_message(message["text"], message["is_user"])
    
    # Message input area
    st.markdown("<div style='padding-bottom: 100px;'></div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <style>
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 20px;
            background-color: white;
            border-top: 1px solid #e0e0e0;
            z-index: 1000;
        }
        </style>
        <div class="input-container">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([6, 1])
        
        with col1:
            query = st.text_input("Ask a question about your documents:", key="query_input", label_visibility="collapsed")
        
        with col2:
            send_button = st.button("Send", type="primary", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Process query and generate response
    if (query or send_button) and st.session_state.data_loaded:
        if query:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "text": query,
                "is_user": True,
                "timestamp": datetime.now()
            })
            
            # Generate AI response
            with st.spinner("Thinking..."):
                response = query_documents(conn, cursor, query)
                
                # Add AI response to chat history
                st.session_state.chat_history.append({
                    "text": response,
                    "is_user": False,
                    "timestamp": datetime.now()
                })
            
            # Clear input and rerun to update display
            st.experimental_rerun()
    elif (query or send_button) and not st.session_state.data_loaded:
        st.warning("Please upload files or provide URLs before asking questions.")
    
    # Close database connection
    if conn:
        conn.close()


if __name__ == "__main__":
    main()#pip install streamlit pandas requests beautifulsoup4 psycopg2-binary google-generativeai PyPDF2 pgvector numpy sentence-transformers langchain python-dotenv selenium webdriver-manager datetime
