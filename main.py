import streamlit as st
from dotenv import load_dotenv
from typing import List
from google.genai import Client
from qdrant_client import QdrantClient
from langchain_voyageai import VoyageAIEmbeddings
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client.models import models
import os
import re
from pathlib import Path

st.set_page_config(page_title="Deakin College Chatbot", layout="centered")
load_dotenv()

class RAGCore:
    def __init__(self):
        print("Running the ragcore")
        self.llm = Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.vc = VoyageAIEmbeddings(model="voyage-3", api_key=os.getenv("VOYAGE_API_KEY"))
        self.qclient = QdrantClient(
            url=os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY"),
            https=True,
            timeout=100,
            # prefer_grpc=True,
        )
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def convert_links_to_markdown(self, text, link_dict):
        def replace_link(match):
            link_text = match.group(1).strip()  # Extract and trim spaces
            normalized_text = " ".join(
                link_text.split()
            )  # Normalize spaces (removes extra spaces)
            url = link_dict.get(normalized_text, "#")  # Lookup in dictionary
            return f"[{normalized_text}]({url})"  # Convert to Markdown link

        # Regex to find <link> text </link> (handling spaces)
        pattern = r"<link>\s*(.*?)\s*</link>"
        return re.sub(pattern, replace_link, text)
    
    # ---- Function: Simulate RAG retrieval (Replace with actual retrieval logic) ----
    @st.cache_resource
    def retrieve_documents(_self,query: str) -> List[str]:
        """Simulate retrieving relevant documents based on a query."""
        meta_dict = {}
        dense_query = _self.vc.embed_query(query)
        sparse_query = next(_self.sparse_embedding_model.query_embed(query))
        # late_query = next(late_iteraction_model.query_embed(query))

        prefetch = [
            models.Prefetch(query=dense_query, using="voyage-3", limit=25),
            models.Prefetch(
                query=models.SparseVector(**sparse_query.as_object()),
                using="bm25",
                limit=25,
            ),
        ]
        results = _self.qclient.query_points(
            "hybrid-search",
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=10,
        )

        list_retrieved_docs = []

        for point in results.points:
            list_retrieved_docs.append(point.payload["document"])
            meta_dict.update(point.payload["metadata"])

        return (list_retrieved_docs, meta_dict)
    
    # ---- Function: Simulate Response Generation (Replace with actual AI model) ----
    @st.cache_resource
    def generate_response(_self,query: str, retrieved_docs: List[str]) -> str:
        """Simulate AI-generated response using retrieved documents."""
        context = ""
        for idx, points in enumerate(retrieved_docs):
            context += points

        prompt = f"""
            System Message:

            You are an creative AI assistant in a student-facing application.

            Answer user queries using only the information provided in the context below. Your goal is to deliver clear, comprehensive, and student-friendly responses. Where appropriate, include explanations that help the student understand the background, reasoning, or implications of the information.
            Answering Guidelines:
            Be thorough and descriptive. Include all relevant details from the context. When applicable, expand on key points to help the student better understand the subject.
            Define or briefly explain technical terms or concepts that may be unfamiliar to a student.
            If the question involves a process, policy, or option, describe each step or component clearly.
            Use structured formatting to improve clarity:
            Numbered steps for procedures or processes
            Bullet points for lists, key points, or alternatives
            Tables for comparisons, categories, or structured data
            Constraints:
            Use only the provided context. Do not guess or invent information.
            If the context does not fully answer the question, state this clearly and suggest that the student contact the appropriate university office or representative.
            Do not modify or omit any <link></link> tags. Include them exactly as provided.
            Always end your answer with a recommendation to consult the most relevant office or person (such as an academic advisor, registrar, financial aid, or IT helpdesk), based on the topic of the query.

            Context:
            {context}

            User Query:
            {query}
        """

        resp = _self.llm.models.generate_content(model="gemini-2.0-flash", contents=[prompt])
        return resp.text

@st.cache_resource   
def create_instance():
    return RAGCore()

rag = create_instance()

# Create three columns, with the image in the center column
col1, col2, col3 = st.columns([1,2, 1])
chat_history = []
st.logo("DeakinCollege.png", link="https://www.deakincollege.edu.au/", size="large")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.chat_history == []:
    with col2:
        st.image("DeakinCollege.png", width=300)
    st.markdown("<h1 style='text-align: center;'  >What can I help you with?</h1>", unsafe_allow_html=True)
    initial_input = st.text_input(label="",placeholder="Get quick answer about your queries...", label_visibility="collapsed")
    if initial_input:
        st.session_state.chat_history.append({
            'role':'user',
            'content': initial_input
        })
        st.rerun()
    st.stop()

st.header("Deakin College Chatbot", divider="gray")
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        if message['role'] == 'bot':
            with st.chat_message('ai'):
                st.write(message['content'])
            with st.expander("📄 Reference Documents"):
                    for doc in message["references"]:
                        st.markdown(f"- {doc}")
    

query = st.chat_input()
if query:
    print(query)
    st.session_state.chat_history.append({
        'role':'user',
        'content':query
    })
    (retrieved_docs, meta_dict) = rag.retrieve_documents(query)

    response = rag.generate_response(query, retrieved_docs)
    final_response = rag.convert_links_to_markdown(response, meta_dict)
    st.session_state.chat_history.append({
        'role':'bot',
        'content': response,
        'references':retrieved_docs,
    })
    # includes all the LLM calling process
    st.rerun()



# st.("*Deakin College Chatbot - Powered by RAG* 🚀")
# with chat_container:
