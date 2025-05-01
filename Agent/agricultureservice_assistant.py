import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import sqlite3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re
import time


class AgricultureServiceAssistant:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self._setup_database()
        self._setup_vector_store()
        self._setup_llm()
        self._setup_rag_chain()

    def _setup_database(self):
        """Initialize the SQL database connection"""
        DATABASE_PATH = "Project_AgroFarmServ_grouop_4.db"
        engine = create_engine(f"sqlite:///{DATABASE_PATH}")
        self.db = SQLDatabase(engine)
        self.schema_text = self._extract_schema_info()

    def _extract_schema_info(self):
        """Extract database schema information"""
        conn = sqlite3.connect("Project_AgroFarmServ_grouop_4.db")
        cursor = conn.cursor()
        schema_info = []

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [f"{col[1]} ({col[2]})" for col in cursor.fetchall()]
            cursor.execute(f"PRAGMA foreign_key_list({table});")
            fks = [f"{fk[3]} → {fk[2]}.{fk[4]}" for fk in cursor.fetchall()]
            info = f"TABLE: {table}\nCOLUMNS: {', '.join(columns)}"
            if fks:
                info += f"\nRELATIONSHIPS: {', '.join(fks)}"
            schema_info.append(info)

        conn.close()
        return "\n\n".join(schema_info)

    def _setup_vector_store(self):
        """Initialize Pinecone vector store"""
        pinecone_api_key = os.environ["PINECONE_API_KEY"]
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "agrifarmrag"

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                index_name,
                dimension=1536,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            time.sleep(30)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = PineconeVectorStore.from_documents(
            self._get_documents(),
            embedding=embeddings,
            index_name=index_name,
            namespace="agriculture"
        )

    def _get_documents(self):
        """Prepare documents for vector store"""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [Document(page_content=self.schema_text, metadata={"source": "schema"})] + self._load_table_samples()
        return splitter.split_documents(docs)

    def _load_table_samples(self, sample_size=3):
        """Load sample data from key tables"""
        docs = []
        key_tables = ['service', 'user_service', 'user', 'user_order', 'order_service']

        for table in key_tables:
            try:
                schema = self.db.run(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
                sample = self.db.run(f"SELECT * FROM {table} LIMIT {sample_size};")
                docs.append(Document(
                    page_content=f"TABLE: {table}\nSCHEMA: {schema}\nSAMPLE DATA:\n{sample}",
                    metadata={"table": table}
                ))
            except Exception as e:
                print(f"Error loading {table}: {e}")
        return docs

    def _setup_llm(self):
        """Initialize the language model"""
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    def _setup_rag_chain(self):
        """Set up the RAG chain"""
        template = """
        # You are a multilingual agricultural services assistant. Follow these rules:

        ## Role-Based Access Control
        User Role: {role}
        Access Rules:
        - ADMIN → Full access to all data
        - FARMER → Service provider details (including contact info)
        - PROVIDER → Farmer contacts only (name, email, phone, location)
        - BOTH → Combined access

        ## Multilingual Support
        ALWAYS respond in this language: {language}

        ## Database Query Requirements
        For question: {question}
        - Internally generate optimized SQL by joining the appropriate tables.
        - DO NOT return or show any SQL code in the response.
        - Never say "I need to connect tables" - just do it.
        - user
        - service
        - user_service  
        - user_order
        - order_service
        - payments (admin only)

        ## Response Format
        1. [Accurate answer with joined data]
        2. [Additional context from related tables]
        3. [Personalized recommendations if farmer]
        4. [Suggested next steps]
       

        Database Schema:
        {schema}

        Relevant Context:
        {context}

        Guidelines:
        - Always maintain role-based data privacy
        - Provide complete information without "I would need to join tables"
        - Include prices, availability, and service details
        - For farmers: suggest complementary services
        """
        prompt = ChatPromptTemplate.from_template(template)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        # self.rag_chain = (
        #     {
        #         "context": RunnableLambda(lambda x: retriever.invoke(x["question"])),
        #         "question": RunnablePassthrough(),
        #         "schema": lambda x: self.schema_text,
        #         "role": lambda x: x.get("role", "farmer"),
        #         "language": lambda x: x.get("detected_lang", "en"),
        #         "detected_lang": lambda x: x.get("detected_lang", "en")
        #     }
        #     | prompt
        #     | self.llm
        #     | StrOutputParser()
        # )
        self.rag_chain = (
                {
                    "context": RunnableLambda(lambda x: self._get_relevant_context(x["question"], x["detected_lang"])),
                    "question": RunnablePassthrough(),
                    "schema": lambda x: self.schema_text,
                    "role": lambda x: x.get("role", "farmer"),
                    "language": lambda x: x.get("detected_lang", "en")  # Single language parameter
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )

    def _get_relevant_context(self, question, lang):
        """Retrieve context considering language"""
        try:
            # For English queries, use normal retrieval
            if lang == 'en':
                return self.vector_store.as_retriever().invoke(question)

            # For other languages, translate query to English for better retrieval
            translated_query = self.translate_text(question, 'en')
            return self.vector_store.as_retriever().invoke(translated_query)
        except Exception as e:
            print(f"Context retrieval error: {e}")
            return []

    def detect_language(self, text):
        """Detect the language of input text"""
        prompt = f"Detect the language of this text and respond with only the ISO 639-1 language code:\n\n{text}"
        return self.llm.invoke(prompt).content.strip().lower()

    def translate_text(self, text, target_lang):
        """Translate text to target language"""
        prompt = f"Translate this agricultural service text to {target_lang}:\n- Maintain technical terms\n- Keep farmer-friendly tone\n- Preserve all numbers/units exactly\n\nText:\n{text}\n\nReturn ONLY the translation:"
        return self.llm.invoke(prompt).content.strip()

    def detect_user_role(self, text):
        """Detect user role from input text"""
        # Check for explicit role mentions
        role_keywords = {
            'admin': ['admin', 'administrator'],
            'farmer': ['farmer', 'cultivator', 'grower'],
            'provider': ['provider', 'service provider', 'contractor'],
            'both': ['both', 'farmer and provider']
        }

        input_lower = text.lower()
        for role, keywords in role_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                return role

        # Try to match user by name
        match = re.search(r"([A-Za-z]+ [A-Za-z]+)", text)
        name = match.group(1) if match else None

        if name:
            try:
                query = f"SELECT role FROM user WHERE name LIKE '%{name}%' LIMIT 1;"
                result = self.db.run(query)
                if result and isinstance(result, list):
                    return result[0]['role'].lower()
            except Exception as e:
                print(f"Role detection error: {e}")

        return "farmer"  # Default role

    def handle_query(self, user_input, language='en'):
        """Process user query with role-based access control"""
        try:
            # Step 1: Detect user role
            role = self.detect_user_role(user_input)

            # Step 2: Detect language if auto
            if language == 'auto':
                detected_lang = self.detect_language(user_input)
            else:
                detected_lang = language

            # Step 3: Translate to English if needed
            if detected_lang != 'en':
                english_question = self.translate_text(user_input, 'en')
            else:
                english_question = user_input

            # Step 4: Run RAG chain
            response = self.rag_chain.invoke({
                "question": english_question,
                "role": role,
                "language": language,
                "detected_lang": detected_lang
            })

            # Step 5: Translate back if needed
            if language != 'en':
                response = self.translate_text(response, language)

            return response

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            if language != 'en':
                return self.translate_text(error_msg, language)
            return error_msg