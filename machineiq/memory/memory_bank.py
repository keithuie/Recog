"""SQLite-based memory storage for patterns and machine data."""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import numpy as np


@dataclass
class PatternRecord:
    """Record of a stored pattern.

    Attributes:
        id: Unique identifier (auto-generated).
        machine_id: ID of the associated machine.
        timestamp: When the pattern was recorded.
        feature_vector: The feature values as numpy array.
        is_anomaly: Whether this was flagged as an anomaly.
        anomaly_score: Score from detection (0-1).
        label: Optional label (e.g., "normal", "bearing_fault").
        metadata: Additional metadata as dictionary.
    """
    machine_id: str
    timestamp: datetime
    feature_vector: np.ndarray
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    label: Optional[str] = None
    metadata: Optional[Dict] = None
    id: Optional[int] = None

    def to_row(self) -> Tuple:
        """Convert to database row tuple."""
        return (
            self.machine_id,
            self.timestamp.isoformat(),
            self.feature_vector.tobytes(),
            self.feature_vector.dtype.str,
            len(self.feature_vector),
            int(self.is_anomaly),
            self.anomaly_score,
            self.label,
            json.dumps(self.metadata) if self.metadata else None,
        )

    @classmethod
    def from_row(cls, row: Tuple) -> "PatternRecord":
        """Create from database row."""
        id_, machine_id, ts, vec_bytes, dtype, size, is_anom, score, label, meta = row
        feature_vector = np.frombuffer(vec_bytes, dtype=dtype).copy()
        return cls(
            id=id_,
            machine_id=machine_id,
            timestamp=datetime.fromisoformat(ts),
            feature_vector=feature_vector,
            is_anomaly=bool(is_anom),
            anomaly_score=score,
            label=label,
            metadata=json.loads(meta) if meta else None,
        )


@dataclass
class MachineRecord:
    """Record of a registered machine.

    Attributes:
        id: Unique machine identifier.
        name: Human-readable machine name.
        description: Optional description.
        config: Machine-specific configuration.
        created_at: When the machine was registered.
        updated_at: When the record was last updated.
    """
    id: str
    name: str
    description: Optional[str] = None
    config: Dict = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_row(self) -> Tuple:
        """Convert to database row tuple."""
        now = datetime.now()
        return (
            self.id,
            self.name,
            self.description,
            json.dumps(self.config),
            (self.created_at or now).isoformat(),
            now.isoformat(),
        )

    @classmethod
    def from_row(cls, row: Tuple) -> "MachineRecord":
        """Create from database row."""
        id_, name, desc, config, created, updated = row
        return cls(
            id=id_,
            name=name,
            description=desc,
            config=json.loads(config) if config else {},
            created_at=datetime.fromisoformat(created),
            updated_at=datetime.fromisoformat(updated),
        )


class MemoryBank:
    """SQLite-based storage for patterns and machine data.

    Stores:
    - Learned normal patterns (for training)
    - Detected anomalies (for labeling and review)
    - Labeled fault patterns (for pattern matching)
    - Machine configuration and metadata

    Thread Safety:
        Each connection is thread-local. Use separate MemoryBank instances
        for multi-threaded access, or use the context manager.
    """

    def __init__(self, db_path: str = ":memory:"):
        """Initialize the memory bank.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory.
        """
        self.db_path = db_path
        self._is_memory = db_path == ":memory:"
        self._conn: Optional[sqlite3.Connection] = None

        # Create parent directories if needed
        if not self._is_memory:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # For in-memory databases, keep a persistent connection
        if self._is_memory:
            self._conn = sqlite3.connect(":memory:")

        # Initialize database schema
        self._init_schema()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with context manager."""
        if self._is_memory:
            # Use persistent connection for in-memory databases
            yield self._conn
            self._conn.commit()
        else:
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    machine_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    feature_vector BLOB NOT NULL,
                    dtype TEXT NOT NULL,
                    vector_size INTEGER NOT NULL,
                    is_anomaly INTEGER DEFAULT 0,
                    anomaly_score REAL DEFAULT 0.0,
                    label TEXT,
                    metadata TEXT,
                    FOREIGN KEY (machine_id) REFERENCES machines(id)
                )
            """)

            # Machines table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS machines (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    config TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_machine
                ON patterns(machine_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_timestamp
                ON patterns(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_label
                ON patterns(label)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_anomaly
                ON patterns(is_anomaly)
            """)

    # Machine operations

    def register_machine(self, machine: MachineRecord) -> None:
        """Register or update a machine.

        Args:
            machine: MachineRecord to store.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO machines (id, name, description, config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, machine.to_row())

    def get_machine(self, machine_id: str) -> Optional[MachineRecord]:
        """Get a machine by ID.

        Args:
            machine_id: Machine identifier.

        Returns:
            MachineRecord if found, None otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM machines WHERE id = ?", (machine_id,))
            row = cursor.fetchone()
            return MachineRecord.from_row(row) if row else None

    def list_machines(self) -> List[MachineRecord]:
        """List all registered machines.

        Returns:
            List of MachineRecord objects.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM machines ORDER BY name")
            return [MachineRecord.from_row(row) for row in cursor.fetchall()]

    def delete_machine(self, machine_id: str) -> bool:
        """Delete a machine and all its patterns.

        Args:
            machine_id: Machine identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Delete patterns first
            cursor.execute("DELETE FROM patterns WHERE machine_id = ?", (machine_id,))

            # Delete machine
            cursor.execute("DELETE FROM machines WHERE id = ?", (machine_id,))

            return cursor.rowcount > 0

    # Pattern operations

    def store_pattern(self, pattern: PatternRecord) -> int:
        """Store a single pattern.

        Args:
            pattern: PatternRecord to store.

        Returns:
            ID of the stored pattern.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO patterns
                (machine_id, timestamp, feature_vector, dtype, vector_size, is_anomaly, anomaly_score, label, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, pattern.to_row())
            return cursor.lastrowid

    def store_patterns(self, patterns: List[PatternRecord]) -> List[int]:
        """Store multiple patterns efficiently.

        Args:
            patterns: List of PatternRecord objects.

        Returns:
            List of stored pattern IDs.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            ids = []
            for pattern in patterns:
                cursor.execute("""
                    INSERT INTO patterns
                    (machine_id, timestamp, feature_vector, dtype, vector_size, is_anomaly, anomaly_score, label, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, pattern.to_row())
                ids.append(cursor.lastrowid)
            return ids

    def get_pattern(self, pattern_id: int) -> Optional[PatternRecord]:
        """Get a pattern by ID.

        Args:
            pattern_id: Pattern identifier.

        Returns:
            PatternRecord if found, None otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patterns WHERE id = ?", (pattern_id,))
            row = cursor.fetchone()
            return PatternRecord.from_row(row) if row else None

    def get_patterns(
        self,
        machine_id: Optional[str] = None,
        label: Optional[str] = None,
        is_anomaly: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[PatternRecord]:
        """Query patterns with filters.

        Args:
            machine_id: Filter by machine ID.
            label: Filter by label.
            is_anomaly: Filter by anomaly status.
            start_time: Filter by minimum timestamp.
            end_time: Filter by maximum timestamp.
            limit: Maximum number of results.

        Returns:
            List of matching PatternRecord objects.
        """
        query = "SELECT * FROM patterns WHERE 1=1"
        params = []

        if machine_id is not None:
            query += " AND machine_id = ?"
            params.append(machine_id)

        if label is not None:
            query += " AND label = ?"
            params.append(label)

        if is_anomaly is not None:
            query += " AND is_anomaly = ?"
            params.append(int(is_anomaly))

        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp DESC"

        if limit is not None:
            query += f" LIMIT {limit}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [PatternRecord.from_row(row) for row in cursor.fetchall()]

    def get_normal_patterns(
        self, machine_id: str, limit: Optional[int] = None
    ) -> List[PatternRecord]:
        """Get normal (non-anomaly) patterns for a machine.

        Args:
            machine_id: Machine identifier.
            limit: Maximum number of results.

        Returns:
            List of normal PatternRecord objects.
        """
        return self.get_patterns(machine_id=machine_id, is_anomaly=False, limit=limit)

    def get_anomaly_patterns(
        self, machine_id: str, limit: Optional[int] = None
    ) -> List[PatternRecord]:
        """Get anomaly patterns for a machine.

        Args:
            machine_id: Machine identifier.
            limit: Maximum number of results.

        Returns:
            List of anomaly PatternRecord objects.
        """
        return self.get_patterns(machine_id=machine_id, is_anomaly=True, limit=limit)

    def get_labeled_patterns(
        self, machine_id: str, label: str, limit: Optional[int] = None
    ) -> List[PatternRecord]:
        """Get patterns with a specific label.

        Args:
            machine_id: Machine identifier.
            label: Pattern label to filter by.
            limit: Maximum number of results.

        Returns:
            List of labeled PatternRecord objects.
        """
        return self.get_patterns(machine_id=machine_id, label=label, limit=limit)

    def update_pattern_label(self, pattern_id: int, label: str) -> bool:
        """Update the label of a pattern.

        Args:
            pattern_id: Pattern identifier.
            label: New label value.

        Returns:
            True if updated, False if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE patterns SET label = ? WHERE id = ?",
                (label, pattern_id)
            )
            return cursor.rowcount > 0

    def delete_pattern(self, pattern_id: int) -> bool:
        """Delete a pattern.

        Args:
            pattern_id: Pattern identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM patterns WHERE id = ?", (pattern_id,))
            return cursor.rowcount > 0

    def delete_patterns(
        self,
        machine_id: Optional[str] = None,
        label: Optional[str] = None,
        is_anomaly: Optional[bool] = None,
    ) -> int:
        """Delete patterns matching criteria.

        Args:
            machine_id: Filter by machine ID.
            label: Filter by label.
            is_anomaly: Filter by anomaly status.

        Returns:
            Number of deleted patterns.
        """
        query = "DELETE FROM patterns WHERE 1=1"
        params = []

        if machine_id is not None:
            query += " AND machine_id = ?"
            params.append(machine_id)

        if label is not None:
            query += " AND label = ?"
            params.append(label)

        if is_anomaly is not None:
            query += " AND is_anomaly = ?"
            params.append(int(is_anomaly))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount

    # Statistics

    def get_pattern_count(
        self,
        machine_id: Optional[str] = None,
        is_anomaly: Optional[bool] = None,
    ) -> int:
        """Get count of patterns matching criteria.

        Args:
            machine_id: Filter by machine ID.
            is_anomaly: Filter by anomaly status.

        Returns:
            Number of matching patterns.
        """
        query = "SELECT COUNT(*) FROM patterns WHERE 1=1"
        params = []

        if machine_id is not None:
            query += " AND machine_id = ?"
            params.append(machine_id)

        if is_anomaly is not None:
            query += " AND is_anomaly = ?"
            params.append(int(is_anomaly))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()[0]

    def get_label_counts(self, machine_id: str) -> Dict[str, int]:
        """Get count of patterns by label for a machine.

        Args:
            machine_id: Machine identifier.

        Returns:
            Dictionary mapping label to count.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT label, COUNT(*) as count
                FROM patterns
                WHERE machine_id = ?
                GROUP BY label
            """, (machine_id,))
            return {row[0] or "unlabeled": row[1] for row in cursor.fetchall()}

    def get_statistics(self, machine_id: str) -> Dict:
        """Get comprehensive statistics for a machine.

        Args:
            machine_id: Machine identifier.

        Returns:
            Dictionary with pattern statistics.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total count
            cursor.execute(
                "SELECT COUNT(*) FROM patterns WHERE machine_id = ?",
                (machine_id,)
            )
            total = cursor.fetchone()[0]

            # Anomaly count
            cursor.execute(
                "SELECT COUNT(*) FROM patterns WHERE machine_id = ? AND is_anomaly = 1",
                (machine_id,)
            )
            anomalies = cursor.fetchone()[0]

            # Score statistics
            cursor.execute("""
                SELECT AVG(anomaly_score), MIN(anomaly_score), MAX(anomaly_score)
                FROM patterns WHERE machine_id = ?
            """, (machine_id,))
            score_row = cursor.fetchone()

            # Time range
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp)
                FROM patterns WHERE machine_id = ?
            """, (machine_id,))
            time_row = cursor.fetchone()

            return {
                "total_patterns": total,
                "normal_patterns": total - anomalies,
                "anomaly_patterns": anomalies,
                "anomaly_rate": anomalies / total if total > 0 else 0.0,
                "score_avg": score_row[0] or 0.0,
                "score_min": score_row[1] or 0.0,
                "score_max": score_row[2] or 0.0,
                "first_pattern": time_row[0],
                "last_pattern": time_row[1],
                "label_counts": self.get_label_counts(machine_id),
            }

    # Utility methods

    def get_feature_vectors(
        self,
        machine_id: str,
        is_anomaly: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> np.ndarray:
        """Get feature vectors as a numpy array.

        Convenience method for training detectors.

        Args:
            machine_id: Machine identifier.
            is_anomaly: Filter by anomaly status.
            limit: Maximum number of patterns.

        Returns:
            2D numpy array of shape (n_patterns, n_features).
        """
        patterns = self.get_patterns(
            machine_id=machine_id,
            is_anomaly=is_anomaly,
            limit=limit,
        )

        if not patterns:
            return np.array([])

        return np.array([p.feature_vector for p in patterns])

    def export_to_csv(self, path: str, machine_id: Optional[str] = None) -> int:
        """Export patterns to CSV file.

        Args:
            path: Output file path.
            machine_id: Optional filter by machine.

        Returns:
            Number of exported patterns.
        """
        import csv

        patterns = self.get_patterns(machine_id=machine_id)
        if not patterns:
            return 0

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            n_features = len(patterns[0].feature_vector)
            header = ['id', 'machine_id', 'timestamp', 'is_anomaly', 'anomaly_score', 'label']
            header.extend([f'feature_{i}' for i in range(n_features)])
            writer.writerow(header)

            # Data
            for p in patterns:
                row = [p.id, p.machine_id, p.timestamp.isoformat(), p.is_anomaly, p.anomaly_score, p.label]
                row.extend(p.feature_vector.tolist())
                writer.writerow(row)

        return len(patterns)

    def close(self) -> None:
        """Close any open connections.

        Note: Connections are automatically closed by context manager,
        but this can be called for explicit cleanup.
        """
        pass  # Connections are managed per-operation
