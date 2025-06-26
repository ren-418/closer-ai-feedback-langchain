import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_CLOUD = os.getenv('PINECONE_CLOUD', 'aws')
PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')


pc = Pinecone(api_key=PINECONE_API_KEY)

class PineconeManager:
    def __init__(self, index_name: str = "sales-calls"):
        """Initialize Pinecone manager with specified index using new Pinecone client API."""
        self.index_name = index_name
        # Create index if it doesn't exist
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
        self.index = pc.Index(index_name)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI text-embedding-3-large."""
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding

    def store_transcript(self, transcript_text: str, metadata: Dict[str, Any]) -> str:
        """
        Store transcript embedding in Pinecone with minimal metadata (filename, closer_name, date, transcript).
        Returns the vector ID.
        """
        vector_id = metadata.get('filename', str(hash(transcript_text)))
        embedding = self.generate_embedding(transcript_text)
        # Upsert the vector with minimal metadata
        self.index.upsert(vectors=[
            (vector_id, embedding, metadata)
        ])
        return vector_id

    def find_similar_calls(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        Find similar calls using the query text.
        Returns list of (id, score, metadata) tuples.
        """
        query_embedding = self.generate_embedding(query_text)
        # Query the index
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [
            {
                'id': match.id,
                'score': match.score,
                'metadata': match.metadata
            }
            for match in results.matches
        ]

    def delete_vector(self, vector_id: str) -> None:
        """Delete a vector by ID."""
        self.index.delete(ids=[vector_id])

    def get_index_stats(self) -> Dict:
        """Get statistics about the index."""
        return self.index.describe_index_stats() 