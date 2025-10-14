"""
Training Data Collection System - Storage Manager
=================================================

Manages persistent storage of training trajectories.

Author: MedIllustrator-AI Team
Version: 3.2.0
Date: 2025-10-14
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import logging
from datetime import datetime
import sqlite3
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages persistent storage of training trajectories.
    
    Features:
    - Multiple storage backends (JSON, SQLite, Cloud Storage)
    - Efficient batch operations
    - Metadata indexing
    - Query capabilities
    - Backup and recovery
    
    Storage Backends:
        - local_json: JSON files (default)
        - sqlite: SQLite database
        - cloud_storage: Google Cloud Storage (future)
    
    Example:
        >>> storage = StorageManager(backend="sqlite")
        >>> storage.store_trajectory(trajectory)
        >>> stored = storage.load_trajectory(trajectory_id)
    """
    
    def __init__(
        self,
        backend: str = "local_json",
        storage_path: str = "phase2/training/data",
        db_name: str = "trajectories.db"
    ):
        """
        Initialize storage manager.
        
        Args:
            backend: Storage backend ("local_json" or "sqlite")
            storage_path: Base storage path
            db_name: Database name (for SQLite backend)
        """
        self.backend = backend
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / db_name
        
        # Initialize backend
        if backend == "sqlite":
            self._initialize_sqlite()
        
        logger.info(
            f"StorageManager initialized: backend={backend}, "
            f"path={storage_path}"
        )
    
    def store_trajectory(
        self,
        trajectory: Any,  # AssessmentTrajectory type
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store trajectory to persistent storage.
        
        Args:
            trajectory: Assessment trajectory to store
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.backend == "local_json":
                return self._store_json(trajectory, metadata)
            elif self.backend == "sqlite":
                return self._store_sqlite(trajectory, metadata)
            else:
                logger.error(f"Unsupported backend: {self.backend}")
                return False
        except Exception as e:
            logger.error(
                f"Failed to store trajectory {trajectory.trajectory_id}: {e}"
            )
            return False
    
    def load_trajectory(
        self,
        trajectory_id: str
    ) -> Optional[Any]:
        """
        Load trajectory from storage.
        
        Args:
            trajectory_id: Trajectory identifier
            
        Returns:
            AssessmentTrajectory if found, None otherwise
        """
        try:
            if self.backend == "local_json":
                return self._load_json(trajectory_id)
            elif self.backend == "sqlite":
                return self._load_sqlite(trajectory_id)
            else:
                logger.error(f"Unsupported backend: {self.backend}")
                return None
        except Exception as e:
            logger.error(
                f"Failed to load trajectory {trajectory_id}: {e}"
            )
            return None
    
    def store_batch(
        self,
        trajectories: List[Any]
    ) -> int:
        """
        Store batch of trajectories.
        
        Args:
            trajectories: List of trajectories to store
            
        Returns:
            Number of trajectories successfully stored
        """
        success_count = 0
        
        for trajectory in trajectories:
            if self.store_trajectory(trajectory):
                success_count += 1
        
        logger.info(
            f"Stored batch: {success_count}/{len(trajectories)} successful"
        )
        
        return success_count
    
    def query_trajectories(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[str]:
        """
        Query trajectory IDs based on filters.
        
        Args:
            filters: Filter criteria (e.g., {"session_id": "abc123"})
            limit: Maximum number of results
            
        Returns:
            List of trajectory IDs
        """
        if self.backend == "local_json":
            return self._query_json(filters, limit)
        elif self.backend == "sqlite":
            return self._query_sqlite(filters, limit)
        else:
            logger.error(f"Unsupported backend: {self.backend}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Statistics dictionary
        """
        if self.backend == "local_json":
            return self._stats_json()
        elif self.backend == "sqlite":
            return self._stats_sqlite()
        else:
            return {}
    
    # JSON Backend Implementation
    def _store_json(
        self,
        trajectory: Any,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Store trajectory as JSON file."""
        filename = f"trajectory_{trajectory.trajectory_id}.json"
        filepath = self.storage_path / filename
        
        data = trajectory.to_dict()
        if metadata:
            data["storage_metadata"] = metadata
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Stored trajectory to {filepath}")
        return True
    
    def _load_json(self, trajectory_id: str) -> Optional[Any]:
        """Load trajectory from JSON file."""
        filename = f"trajectory_{trajectory_id}.json"
        filepath = self.storage_path / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct trajectory object
        # Note: This requires importing AssessmentTrajectory
        # For now, return raw data
        return data
    
    def _query_json(
        self,
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> List[str]:
        """Query JSON files."""
        trajectory_ids = []
        
        for filepath in self.storage_path.glob("trajectory_*.json"):
            if len(trajectory_ids) >= limit:
                break
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Apply filters
                if filters:
                    matches = all(
                        data.get(key) == value
                        for key, value in filters.items()
                    )
                    if not matches:
                        continue
                
                trajectory_ids.append(data.get("trajectory_id"))
                
            except Exception as e:
                logger.warning(f"Failed to read {filepath}: {e}")
        
        return trajectory_ids
    
    def _stats_json(self) -> Dict[str, Any]:
        """Get JSON storage statistics."""
        files = list(self.storage_path.glob("trajectory_*.json"))
        
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "backend": "local_json",
            "total_trajectories": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "storage_path": str(self.storage_path)
        }
    
    # SQLite Backend Implementation
    def _initialize_sqlite(self):
        """Initialize SQLite database."""
        with self._get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    timestamp TEXT,
                    image_hash TEXT,
                    duration_seconds REAL,
                    num_states INTEGER,
                    data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id
                ON trajectories(session_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON trajectories(timestamp)
            """)
            
            conn.commit()
        
        logger.debug("SQLite database initialized")
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _store_sqlite(
        self,
        trajectory: Any,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Store trajectory in SQLite."""
        with self._get_db_connection() as conn:
            data_json = trajectory.to_json()
            
            conn.execute("""
                INSERT OR REPLACE INTO trajectories
                (trajectory_id, session_id, timestamp, image_hash,
                 duration_seconds, num_states, data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                trajectory.trajectory_id,
                trajectory.session_id,
                trajectory.timestamp.isoformat(),
                trajectory.image_hash,
                trajectory.metadata.get("duration_seconds", 0),
                len(trajectory.intermediate_states),
                data_json
            ))
            
            conn.commit()
        
        logger.debug(f"Stored trajectory in SQLite: {trajectory.trajectory_id}")
        return True
    
    def _load_sqlite(self, trajectory_id: str) -> Optional[Any]:
        """Load trajectory from SQLite."""
        with self._get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT data FROM trajectories
                WHERE trajectory_id = ?
            """, (trajectory_id,))
            
            row = cursor.fetchone()
            
            if row:
                return json.loads(row["data"])
            
            return None
    
    def _query_sqlite(
        self,
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> List[str]:
        """Query SQLite database."""
        query = "SELECT trajectory_id FROM trajectories"
        params = []
        
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"{key} = ?")
                params.append(value)
            
            query += " WHERE " + " AND ".join(conditions)
        
        query += f" LIMIT {limit}"
        
        with self._get_db_connection() as conn:
            cursor = conn.execute(query, params)
            return [row["trajectory_id"] for row in cursor.fetchall()]
    
    def _stats_sqlite(self) -> Dict[str, Any]:
        """Get SQLite storage statistics."""
        with self._get_db_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM trajectories")
            count = cursor.fetchone()["count"]
            
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                "backend": "sqlite",
                "total_trajectories": count,
                "database_size_bytes": db_size,
                "database_size_mb": db_size / (1024 * 1024),
                "database_path": str(self.db_path)
            }


# Finish
