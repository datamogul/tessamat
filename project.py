# -*- coding: utf-8 -*-

import os
import streamlit as st
import pandas as pd
import torch
from dotenv import load_dotenv
from datasets import load_dataset

# Haystack Komponenten laden
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator 
from haystack.utils import Secret
# --- 1. Schrit: Variablen und API Keys laden ---

try:
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY nicht gefunden. .env oder Streamlit secrets kontrollieren")
        st.stop()
except Exception as e:
    st.error(f"Fehler-kontrolliere API-Keys: {e}")
    st.stop()

# --- 2. Schritt Vorbereitung der Daten
# Streamlit cache aktivieren
@st.cache_resource
def load_and_prepare_data():
    """
    
    """
    with st.spinner("LOUIS-Kursdaten werden geladen..."):
        try:
            csv_file_path = "louis_rag_dataset_csv_nov2025.csv" # delimiter ';' ja-nein 
            df = pd.read_csv(csv_file_path, sep=";")
            
            # 'rag_text' leere Felder raus
            df_louis = df[df['rag_text'].notna() & (df['rag_text']!= '')].copy()
            df_louis.reset_index(drop=True, inplace=True)


            # Haystack Objekte kreieren
            documents = list()
            for _, row in df_louis.iterrows():
                content = f"{row['anreisser_louis']}\n\n{row['rag_text']}"
                meta = {
                    'produkttitel': str(row['produkttitel']),
                    'webseite': str(row['url_louis_website']),
                    'produktnummer': int(row['produktnummer']) if pd.notna(row['produktnummer']) else None,
                    'themenfelder': str(row['themenfelder']),
                    
                    # NEU und empfohlen:
                    'zielgruppen': str(row['zielgruppen']),
                    'bildungsprodukttyp': str(row['bildungsprodukttyp']),
                    'foerdermoeglichkeiten': str(row['foerdermoeglichkeiten']),
                    'abschlussarten': str(row['abschlussarten']),
                    'voraussetzungen': str(row['voraussetzungen_louis'])
                }
                documents.append(Document(content=content, meta=meta))
            
            # Splitten der Dokumente
            splitter = DocumentSplitter(split_by="word", split_length=400, split_overlap=75)
            split_docs = splitter.run(documents)
            
            return split_docs['documents']
        except Exception as e:
            st.error(f"Fehler beim Laden des Datasets: {e}")
            return None

# --- 3. Schritt: FAISS VektorDB kreation ---
# Nehme Dokumente und nutze das embedding model
@st.cache_resource
def create_faiss_index(_split_docs):
    """
    Die Dokumente kommen in den InMemory DocumentStore.
    """
    if not _split_docs:
        return None
        
    with st.spinner("VektorDB wird erzeugt..."):
        try:
            document_store = InMemoryDocumentStore()
            
            doc_embedder = SentenceTransformersDocumentEmbedder(
                model="mixedbread-ai/mxbai-embed-large-v1", device=None
            )

            # pipeline für document store
            indexing_pipeline = Pipeline()
            indexing_pipeline.add_component("embedder", doc_embedder)
            indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
            indexing_pipeline.connect("embedder.documents", "writer.documents")

            # erzeuge index 
            indexing_pipeline.run({"embedder": {"documents": _split_docs}})
            
            return document_store
        except Exception as e:
            st.error(f"Fehler beim kreieren des Index: {e}")
            return None

# --- 4. Schritt: RAG Pipeline Aufbau ---
# RAG wird mit allen Komponenten verbunden (retriever, prompt, generator)
# Haystack Pipeline wird generiert.
@st.cache_resource
def build_rag_pipeline(_document_store):
    """
    Der DocumentStore wird verwendetum die RAG Pipeline zu erzeugen.
    """
    if not _document_store:
        return None
        
    try:
        # 1. (Retriever)
        retriever = InMemoryEmbeddingRetriever(document_store=_document_store, top_k=3)
        
        # 2. Prompt Template
       template = """
{% message role="system" %}
Sie sind der spezialisierte Content-Generator für unsere LOUIS Bildungsprodukte. Ihre Hauptaufgabe ist es, die Frage des Benutzers NUR anhand der bereitgestellten KONTEXT-DOKUMENTE zu beantworten.

TONALITÄT UND ROLLE:
- Wenn die Anfrage des Benutzers die Generierung von **Social Media Posts (Instagram, LinkedIn)** betrifft, nutze die **freundliche DU-Form** und antworte im Stil eines engagierten Social-Media-Managers.
- Bei allen anderen Anfragen (z.B. SEO-Texte, Beschreibungen, allgemeine Fragen) verwende die **formelle SIE-Form**.

INHALTSERSTELLUNG:
- Bei der Generierung von **Social Media Posts** (max. 100 Wörter) füge bitte relevante **Hashtags** und einen klaren **Call-to-Action (CTA)** hinzu.
- Bei der Generierung von **SEO-Texten** nutze Zwischenüberschriften (Markdown ##) und eine hohe Keyword-Dichte.

STIL UND LESBARKEIT:
- Stelle sicher, dass der generierte Text eine **hohe Lesbarkeit** aufweist und einen **Flesch-Reading-Ease Score von mindestens 70** erreicht. Verwende kurze Sätze, eine aktive Sprache und klare, einfache Wörter.

INTEGRITÄT DES WISSENS:
- Stützen Sie Ihre Antwort **ausschließlich** auf die bereitgestellten Dokumente und fügen Sie **keine** eigenen Informationen hinzu.
- Wenn die Dokumente nicht genügend Informationen zur Beantwortung der Frage enthalten, antworte **höflich und passend zur verwendeten Tonalität**, dass Du/Sie keine ausreichenden Informationen in den Dokumenten finden konntest/konnten.

QUELLENANGABE:
Führe am Ende jeder Antwort die **Titel und die Webseite** (URL) aller von Ihnen verwendeten Bildungsprodukte unter der Überschrift **'Website:'** auf.

KONTEXT-DOKUMENTE:
{% for doc in documents %}
--- Produkt: {{ doc.meta['produkttitel'] }} ---
Link: {{ doc.meta['webseite'] }}
Inhalt: {{ doc.content }}
{% endfor %}
{% endmessage %}

{% message role="user" %}
Frage: {{question}}
Antwort:
{% endmessage %}
"""
        prompt_builder = ChatPromptBuilder(
            template=template, 
            required_variables=["documents", "question"]
            )

        # 3. Generator
        #message_converter = ChatMessageConverter()
        generator = GoogleGenAIChatGenerator(model="gemini-2.5-flash", api_key=Secret.from_token(GOOGLE_API_KEY))
        # embedding-model für Fragen
        text_embedder = SentenceTransformersTextEmbedder(model="mixedbread-ai/mxbai-embed-large-v1", device=None)

        # 4. RAG Pipeline
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("text_embedder", text_embedder)
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("generator", generator)

        # Komponenten verbinden
        rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder.prompt", "generator.messages") 
        return rag_pipeline
    except Exception as e:
        st.error(f"Fehler beim Erstellen der  RAG Pipeline: {e}")
        return None

# --- 5. Schritt Streamlit Webfrontend ---
def main():
    st.set_page_config(page_title="Tessatestfeld12-11-25", page_icon=":)")
    
    st.title("Tessamat - SEO, LinkedIn & Insta-Postbuilder")
    st.caption("E-Commerce Rulez V1.0.1 merged Dataset louis_rag_dataset_DS_V1_20251112_R1.csv")

    # Komponeten laden und cachen
    split_documents = load_and_prepare_data()
    if split_documents:
        document_store = create_faiss_index(split_documents)
        if document_store:
            rag_pipeline = build_rag_pipeline(document_store)
        else:
            rag_pipeline = None
    else:
        rag_pipeline = None

    if not rag_pipeline:
        st.warning("Konnte Chat nicht starten..")
        st.stop()

    # Verstecke ältere Eingaben mithilfe von session state kullan
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Zeige alte Eingaben
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Nutzereingabe annehmen
    if prompt := st.chat_input("Bsp: Kannst du mir einen LinkedIn Post zur Umschulug Kaufmann im E-Commerce generieren?"):
        # Hinzufügen zum Chatverlauf
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Starte RAG Pipeline und empfange die Antwort
        with st.spinner("Inspiziere Ismails Dataset und erzeuge eine Antwort für Tessa..."):
            try:
                result = rag_pipeline.run({
                    "text_embedder": {"text": prompt},
                    "prompt_builder": {"question": prompt}
                })
                
                response = "Fehler - Konnte keine Antwort empfangen."
                if result and "generator" in result and result["generator"]["replies"]:
                    chat_message = result["generator"]["replies"][0]
                    response = chat_message.text # <-- KORRIGIERT: Zugriff direkt über .text

            except Exception as e:
                response = f"Bei der Verarbeitung der Frage kam es zu einem Problem: {e}"

        # Antwort in den Chatverlauf hinzufügen und Anzeigen
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
