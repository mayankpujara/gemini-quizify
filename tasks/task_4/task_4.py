# embedding_client.py
import streamlit as st
from langchain_google_vertexai import VertexAIEmbeddings

class EmbeddingClient:
    
    def __init__(self, model_name, project, location):
        
        self.model_name = model_name
        self.project = project
        self.location = location

        self.client = VertexAIEmbeddings(
            model_name, project, location
        )

    def embed_query(self, query):
        """
        Uses the embedding client to retrieve embeddings for the given query.
        """
        vectors = self.client.embed_query(query)
        return vectors

    def embed_documents(self, documents):
        """
        Retrieve embeddings for multiple documents.
        """
        try:
            return self.client.embed_documents(documents)
        except AttributeError:
            print("Method embed_documents not defined for the client.")
            return None

if __name__ == "__main__":
    model_name = "textembedding-gecko@003"
    project = "radicalx-quizzify"
    location = "us-central1"

    embedding_client = EmbeddingClient(model_name, project, location)
    vectors = embedding_client.embed_query("Hello World!")
    if vectors:
        st.write(vectors)
        print(vectors)
        st.write("Successfully used the embedding client!")
        print("Successfully used the embedding client!")
