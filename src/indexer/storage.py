"""FAISS index and metadata state management code."""

import os
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss

from ..utils import get_logger, ensure_directory_exists

logger = get_logger(__name__)


class MetadataStore:
    """Metadata storage using SQLite for structured records."""
    
    def __init__(self, db_path: str):
        """
        Initialize metadata store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        ensure_directory_exists(os.path.dirname(db_path))
        self._initialize_schema()
        
    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                caption TEXT,
                attributes TEXT,
                image_embedding BLOB,
                caption_embedding BLOB,
                fashionpedia_image_id INTEGER,
                has_fashionpedia_annotations INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_path 
            ON metadata (file_path)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fashionpedia_image_id 
            ON metadata (fashionpedia_image_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_has_fashionpedia 
            ON metadata (has_fashionpedia_annotations)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Initialized metadata store at {self.db_path}")
        
    def add_record(
        self,
        file_path: str,
        caption: str,
        attributes: Dict[str, Any],
        image_embedding: np.ndarray,
        caption_embedding: np.ndarray,
        fashionpedia_image_id: Optional[int] = None,
        has_fashionpedia_annotations: bool = False
    ) -> int:
        """
        Add a new metadata record.
        
        Args:
            file_path: Path to image file
            caption: Generated caption
            attributes: Parsed attributes JSON
            image_embedding: 512-d image embedding
            caption_embedding: 512-d caption embedding
            fashionpedia_image_id: Original Fashionpedia image ID
            has_fashionpedia_annotations: Whether Fashionpedia annotations exist
            
        Returns:
            Generated image_id
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            image_bytes = image_embedding.astype(np.float32).tobytes()
            caption_bytes = caption_embedding.astype(np.float32).tobytes()
            
            cursor.execute("""
                INSERT INTO metadata (
                    file_path, caption, attributes,
                    image_embedding, caption_embedding,
                    fashionpedia_image_id, has_fashionpedia_annotations
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                file_path,
                caption,
                json.dumps(attributes),
                image_bytes,
                caption_bytes,
                fashionpedia_image_id,
                1 if has_fashionpedia_annotations else 0
            ))
            
            image_id = cursor.lastrowid
            conn.commit()
            
            logger.debug(f"Added record for {file_path} with ID {image_id}")
            return image_id
            
        except sqlite3.IntegrityError:
            cursor.execute(
                "SELECT image_id FROM metadata WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            if row:
                image_id = row[0]
                logger.debug(f"Record already exists for {file_path} with ID {image_id}")
                return image_id
            raise
        finally:
            conn.close()
            
    def get_record(self, image_id: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata record by image_id.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Record dictionary or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT image_id, file_path, caption, attributes,
                   image_embedding, caption_embedding,
                   fashionpedia_image_id, has_fashionpedia_annotations,
                   created_at
            FROM metadata
            WHERE image_id = ?
        """, (image_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
            
        return {
            "image_id": row[0],
            "file_path": row[1],
            "caption": row[2],
            "attributes": json.loads(row[3]) if row[3] else {},
            "image_embedding": np.frombuffer(row[4], dtype=np.float32) if row[4] else None,
            "caption_embedding": np.frombuffer(row[5], dtype=np.float32) if row[5] else None,
            "fashionpedia_image_id": row[6],
            "has_fashionpedia_annotations": bool(row[7]),
            "created_at": row[8]
        }
        
    def get_record_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata record by file path.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Record dictionary or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT image_id, file_path, caption, attributes,
                   image_embedding, caption_embedding,
                   fashionpedia_image_id, has_fashionpedia_annotations,
                   created_at
            FROM metadata
            WHERE file_path = ?
        """, (file_path,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
            
        return {
            "image_id": row[0],
            "file_path": row[1],
            "caption": row[2],
            "attributes": json.loads(row[3]) if row[3] else {},
            "image_embedding": np.frombuffer(row[4], dtype=np.float32) if row[4] else None,
            "caption_embedding": np.frombuffer(row[5], dtype=np.float32) if row[5] else None,
            "fashionpedia_image_id": row[6],
            "has_fashionpedia_annotations": bool(row[7]),
            "created_at": row[8]
        }
    
    def get_records_by_fashionpedia_id(self, fashionpedia_image_id: int) -> List[Dict[str, Any]]:
        """
        Get all metadata records by Fashionpedia image ID.
        
        Args:
            fashionpedia_image_id: Fashionpedia image ID
            
        Returns:
            List of record dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT image_id, file_path, caption, attributes,
                   image_embedding, caption_embedding,
                   fashionpedia_image_id, has_fashionpedia_annotations,
                   created_at
            FROM metadata
            WHERE fashionpedia_image_id = ?
        """, (fashionpedia_image_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            records.append({
                "image_id": row[0],
                "file_path": row[1],
                "caption": row[2],
                "attributes": json.loads(row[3]) if row[3] else {},
                "image_embedding": np.frombuffer(row[4], dtype=np.float32) if row[4] else None,
                "caption_embedding": np.frombuffer(row[5], dtype=np.float32) if row[5] else None,
                "fashionpedia_image_id": row[6],
                "has_fashionpedia_annotations": bool(row[7]),
                "created_at": row[8]
            })
        
        return records
    
    def get_all_records(self) -> List[Dict[str, Any]]:
        """
        Get all metadata records.
        
        Returns:
            List of record dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT image_id, file_path, caption, attributes,
                   image_embedding, caption_embedding,
                   fashionpedia_image_id, has_fashionpedia_annotations,
                   created_at
            FROM metadata
            ORDER BY image_id
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            records.append({
                "image_id": row[0],
                "file_path": row[1],
                "caption": row[2],
                "attributes": json.loads(row[3]) if row[3] else {},
                "image_embedding": np.frombuffer(row[4], dtype=np.float32) if row[4] else None,
                "caption_embedding": np.frombuffer(row[5], dtype=np.float32) if row[5] else None,
                "fashionpedia_image_id": row[6],
                "has_fashionpedia_annotations": bool(row[7]),
                "created_at": row[8]
            })
        
        return records
    
    def get_count(self) -> int:
        """Get total number of records."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM metadata")
        count = cursor.fetchone()[0]
        conn.close()
        return count


class VectorIndex:
    """FAISS vector index manager for efficient similarity search."""
    
    def __init__(
        self,
        index_path: str,
        dimension: int = 512,
        index_type: str = "FlatL2",
        metric: str = "L2"
    ):
        """
        Initialize vector index.
        
        Args:
            index_path: Path to save/load FAISS index
            dimension: Vector dimension
            index_type: Type of index (FlatL2, IVFFlat)
            metric: Distance metric (L2, IP)
        """
        self.index_path = index_path
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        
        ensure_directory_exists(os.path.dirname(index_path))
        self._initialize_index()
        self._id_mapping = []
        
    def _initialize_index(self) -> None:
        """Initialize or load FAISS index."""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded existing index from {self.index_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load index: {e}, creating new")
        
        if self.index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        logger.info(f"Created new {self.index_type} index with dimension {self.dimension}")
        
    def add_embedding(self, image_id: int, embedding: np.ndarray) -> None:
        """
        Add embedding to index.
        
        Args:
            image_id: Image identifier
            embedding: 512-d embedding vector
        """
        embedding = np.array(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        if self.index_type == "IVFFlat" and not self.index.is_trained:
            self.index.train(embedding)
            
        self.index.add(embedding)
        self._id_mapping.append(image_id)
        
    def add_embeddings(self, image_ids: List[int], embeddings: np.ndarray) -> None:
        """
        Add multiple embeddings to index.
        
        Args:
            image_ids: List of image identifiers
            embeddings: Array of embeddings with shape (n, 512)
        """
        embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D array")
            
        if self.index_type == "IVFFlat" and not self.index.is_trained:
            self.index.train(embeddings)
            
        self.index.add(embeddings)
        self._id_mapping.extend(image_ids)
        
    def search(self, query_embedding: np.ndarray, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        k = min(k, self.index.ntotal) if self.index.ntotal > 0 else 0
        if k == 0:
            return np.array([]), np.array([])
            
        distances, indices = self.index.search(query_embedding, k)
        return distances[0], indices[0]
        
    def get_image_id_at_index(self, index_position: int) -> Optional[int]:
        """
        Get image_id at a specific index position.
        
        Args:
            index_position: Position in the index
            
        Returns:
            image_id or None if out of bounds
        """
        if 0 <= index_position < len(self._id_mapping):
            return self._id_mapping[index_position]
        return None
        
    def get_image_ids_by_indices(self, indices: np.ndarray) -> List[int]:
        """
        Get image_ids for a list of index positions.
        
        Args:
            indices: Array of index positions
            
        Returns:
            List of image_ids
        """
        return [self._id_mapping[idx] for idx in indices if 0 <= idx < len(self._id_mapping)]
        
    def save(self) -> None:
        """Save index to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved index to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
            
    def get_total_vectors(self) -> int:
        """Get total number of vectors in index."""
        return self.index.ntotal