import pandas as pd
import chromadb
import uuid
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class Portfolio:
    def __init__(self, file_path="app/resource/my_portfolio.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

        # ✅ Use HuggingFace-based embedding function instead of ONNX
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        # ✅ Explicitly specify tenant and database
        self.chroma_client = chromadb.PersistentClient(
            path="vectorstore",
            tenant="default_tenant",
            database="default_database"
        )

        # Create or connect to collection with the custom embedding function
        self.collection = self.chroma_client.get_or_create_collection(
            name="portfolio",
            embedding_function=embedding_fn
        )

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                techstack = str(row.get("Techstack", "")).strip()
                links = str(row.get("Links", "")).strip()

                if techstack:  # Only add non-empty techstack rows
                    self.collection.add(
                        documents=[techstack],
                        metadatas={"links": links},
                        ids=[str(uuid.uuid4())]
                    )

    def query_links(self, skills):
        if not skills:
            return []

        result = self.collection.query(query_texts=skills, n_results=2)
        metadatas = result.get("metadatas", [])

        # Flatten metadata list and extract links
        links = []
        for meta in metadatas:
            for entry in meta:
                if isinstance(entry, dict) and "links" in entry:
                    links.append(entry["links"])

        return links
