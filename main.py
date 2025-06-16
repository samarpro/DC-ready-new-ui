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
from supabase import create_client
import uuid

st.set_page_config(page_title="Deakin College Chatbot", layout="centered")
load_dotenv()

# removing github icons
hide_toolbar_css = """
<style>
    div.stToolbarActionButton {
        display: none;
    }
</style>
"""

st.markdown(hide_toolbar_css, unsafe_allow_html=True)

def disclaimer_popup():
    if "agreed" not in st.session_state:
        st.session_state.agreed = False

    if not st.session_state.agreed:
        with st.container():
            st.markdown("## ⚠️ Disclaimer")
            st.markdown("""
            This AI chatbot is provided by **Deakin College** as a trial service, powered by AI models and publicly available information.

            **Responses may be inaccurate, incomplete, or biased. Please use your judgment before making any decisions based on chatbot responses.**

            🚫 **Do not input any private, sensitive, or regulated data.**

            Deakin College is **not liable** for any actions, losses, or damages resulting from the use of this chatbot.

            By using this chatbot, you agree that **inputs and outputs may be logged** and used to improve the service.

            For details, refer to Deakin College's digital services policy or contact student support.
            """, unsafe_allow_html=True)

            if st.button("I Agree"):
                st.session_state.agreed = True
                st.rerun()

        # Stop rest of the app from running
        st.stop()

# Call this at the start of your Streamlit app
disclaimer_popup()


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

class RAGCore:
    def __init__(self):
        self.llm = Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.vc = VoyageAIEmbeddings(model="voyage-3", api_key=os.getenv("VOYAGE_API_KEY"))
        self.qclient = QdrantClient(
            url=os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY"),
            https=True,
            timeout=100,
            # prefer_grpc=True,
        )
        self.supabase = create_client(os.getenv("SUPABASE_URL"),os.getenv("SUPABASE_KEY")).table('DC-analysis')
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        self.convo_uuid = None
    

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
    
    def update_feedback(_self, convo_uuid,feedback):
        _self.supabase.update({'feedback':feedback}).eq('uuid_id',convo_uuid).execute()
    
    # ---- Function: Simulate RAG retrieval (Replace with actual retrieval logic) ----
    @st.cache_resource(show_spinner=False)
    def retrieve_documents(_self,query: str) -> List[str]:
        """Simulate retrieving relevant documents based on a query."""
        meta_dict = {}
        # here convo-> conversation, which means when user ask and AI replies. This is counted as one convo.
        # convo_uuid represents this one back and fourth conversation
        _self.convo_uuid = str(uuid.uuid4())
        _self.supabase.insert({
            'uuid_id':_self.convo_uuid,
            'query': query,
            'session_id': st.session_state.session_id
        }).execute()
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

        return (list_retrieved_docs, meta_dict, _self.convo_uuid)
    
    @st.cache_resource(show_spinner=False)
    def perform_query_expansion(_self, query):
        prompt = f"""
            You rewrite student queries to improve retrieval quality in a dense-embedding-based RAG system for an educational institution.

            Your goal is to convert informal, vague, or conversational inputs into clear, content-rich, standalone queries.

            Guidelines:
            Remove vague references like “this,” “it,” or “that” — always be explicit.
            Make the query self-contained and academically phrased.
            Clarify what the student is asking: definition, comparison, explanation, example, process, etc.
            Include any key terms or constraints that are implied.
            Avoid chatbot-style phrases (“Can you tell me…”).
            ➤ Output only the rewritten query.
            Query:
            {query}
            """
        msg =  _self.llm.models.generate_content(model="gemini-2.0-flash", contents=[prompt])
        print("----- Query: ", msg)
        return msg.text

    # ---- Function: Simulate Response Generation (Replace with actual AI model) ----
    @st.cache_resource(show_spinner=False)
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
            If the input is a greeting or casual message (e.g., "hi", "thanks"), respond politely and explain you're here to help with academic questions, and get quick information about Deakin College.
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
        _self.supabase.update({
            'resp': resp.text
        }).eq("uuid_id", _self.convo_uuid).execute()
        return resp.text

@st.cache_resource(show_spinner=False)   
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
        query = rag.perform_query_expansion(initial_input)
        (retrieved_docs, meta_dict, _uuid) = rag.retrieve_documents(query)

        response = rag.generate_response(initial_input, retrieved_docs)
        final_response = rag.convert_links_to_markdown(response, meta_dict)
        st.session_state.chat_history.append({
            'role':'bot',
            'convo_uuid':_uuid,
            'content': final_response,
            'references':retrieved_docs,
        })
        st.rerun()
    st.stop()

st.header("Deakin College Chatbot", divider="gray")
chat_container = st.container()


def broadcast_feedback(_uuid, feedback=None):
    rag.update_feedback(_uuid, st.session_state[f'feedback_{i}'] )

with chat_container:
    for i,message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(message['content'])
        if message['role'] == 'bot':
            with st.chat_message('ai'):
                st.markdown(message['content'])
                feedback = st.feedback(
                    "thumbs",
                    key=f"feedback_{i}",
                    on_change= broadcast_feedback,
                    args=[message['convo_uuid']]
                    )                
                with st.expander("📄 Reference Documents"):
                        for doc in message["references"]:
                            st.markdown(f"- {doc}")
    

query = st.chat_input()
if query:
    st.session_state.chat_history.append({
        'role':'user',
        'content':query
    })
    with st.status("Understanding query...", expanded= True) as status:
        st.write("Updating the UI")
        
        st.write("Getting query vectors and relevant documents")
        query = rag.perform_query_expansion(query)
        (retrieved_docs, meta_dict, _uuid) = rag.retrieve_documents(query)
        st.write("Synthesizing answer (thinking)")
        response = rag.generate_response(query, retrieved_docs)
        final_response = rag.convert_links_to_markdown(response, meta_dict)
        st.write("Getting you the answer...")
        st.session_state.chat_history.append({
            'role':'bot',
            'convo_uuid':_uuid,
            'content': final_response,
            'references':retrieved_docs,
        })

        # includes all the LLM calling process
        st.rerun()



# st.("*Deakin College Chatbot - Powered by RAG* 🚀")
# with chat_container:
