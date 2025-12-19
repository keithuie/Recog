"""Tests for memory bank."""

from datetime import datetime, timedelta
import numpy as np
import pytest

from machineiq.memory.memory_bank import MemoryBank, PatternRecord, MachineRecord


class TestPatternRecord:
    """Tests for PatternRecord class."""

    def test_basic_creation(self):
        """Test basic PatternRecord creation."""
        record = PatternRecord(
            machine_id="machine-1",
            timestamp=datetime.now(),
            feature_vector=np.array([1.0, 2.0, 3.0]),
            is_anomaly=False,
            anomaly_score=0.1,
        )

        assert record.machine_id == "machine-1"
        assert len(record.feature_vector) == 3

    def test_to_row_and_from_row(self):
        """Test row serialization."""
        original = PatternRecord(
            machine_id="machine-1",
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            feature_vector=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            is_anomaly=True,
            anomaly_score=0.75,
            label="fault",
            metadata={"source": "test"},
        )

        row = original.to_row()

        # Simulate adding id from database
        row_with_id = (1,) + row

        restored = PatternRecord.from_row(row_with_id)

        assert restored.id == 1
        assert restored.machine_id == original.machine_id
        assert restored.is_anomaly == original.is_anomaly
        assert np.array_equal(restored.feature_vector, original.feature_vector)


class TestMachineRecord:
    """Tests for MachineRecord class."""

    def test_basic_creation(self):
        """Test basic MachineRecord creation."""
        machine = MachineRecord(
            id="pump-001",
            name="Main Coolant Pump",
            description="Primary cooling system pump",
            config={"type": "centrifugal", "power": 15},
        )

        assert machine.id == "pump-001"
        assert machine.name == "Main Coolant Pump"
        assert machine.config["power"] == 15


class TestMemoryBank:
    """Tests for MemoryBank class."""

    @pytest.fixture
    def memory(self):
        """Create an in-memory database for testing."""
        return MemoryBank(":memory:")

    def test_register_machine(self, memory):
        """Test machine registration."""
        machine = MachineRecord(
            id="test-1",
            name="Test Machine",
            config={"sensors": 4},
        )
        memory.register_machine(machine)

        retrieved = memory.get_machine("test-1")
        assert retrieved is not None
        assert retrieved.id == "test-1"
        assert retrieved.name == "Test Machine"

    def test_list_machines(self, memory):
        """Test listing machines."""
        for i in range(3):
            machine = MachineRecord(id=f"m-{i}", name=f"Machine {i}")
            memory.register_machine(machine)

        machines = memory.list_machines()
        assert len(machines) == 3

    def test_delete_machine(self, memory):
        """Test machine deletion."""
        machine = MachineRecord(id="to-delete", name="Temp Machine")
        memory.register_machine(machine)

        # Add some patterns
        pattern = PatternRecord(
            machine_id="to-delete",
            timestamp=datetime.now(),
            feature_vector=np.array([1, 2, 3]),
        )
        memory.store_pattern(pattern)

        # Delete machine
        result = memory.delete_machine("to-delete")
        assert result is True

        # Machine should be gone
        assert memory.get_machine("to-delete") is None

        # Patterns should be gone too
        patterns = memory.get_patterns(machine_id="to-delete")
        assert len(patterns) == 0

    def test_store_pattern(self, memory):
        """Test pattern storage."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        pattern = PatternRecord(
            machine_id="m1",
            timestamp=datetime.now(),
            feature_vector=np.array([1.0, 2.0, 3.0]),
            is_anomaly=False,
            label="normal",
        )

        pattern_id = memory.store_pattern(pattern)
        assert pattern_id > 0

        retrieved = memory.get_pattern(pattern_id)
        assert retrieved is not None
        assert np.array_equal(retrieved.feature_vector, pattern.feature_vector)

    def test_store_patterns_batch(self, memory):
        """Test batch pattern storage."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        patterns = []
        for i in range(10):
            patterns.append(PatternRecord(
                machine_id="m1",
                timestamp=datetime.now(),
                feature_vector=np.array([i, i+1, i+2]),
            ))

        ids = memory.store_patterns(patterns)
        assert len(ids) == 10

    def test_get_patterns_with_filters(self, memory):
        """Test pattern retrieval with filters."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        # Store mixed patterns
        now = datetime.now()
        for i in range(10):
            pattern = PatternRecord(
                machine_id="m1",
                timestamp=now + timedelta(hours=i),
                feature_vector=np.array([i]),
                is_anomaly=(i % 3 == 0),
                label="anomaly" if i % 3 == 0 else "normal",
            )
            memory.store_pattern(pattern)

        # Filter by anomaly
        anomalies = memory.get_patterns(machine_id="m1", is_anomaly=True)
        assert len(anomalies) == 4  # 0, 3, 6, 9

        # Filter by label
        normals = memory.get_patterns(machine_id="m1", label="normal")
        assert len(normals) == 6

        # Filter by time
        midpoint = now + timedelta(hours=5)
        recent = memory.get_patterns(machine_id="m1", start_time=midpoint)
        assert len(recent) == 5  # hours 5, 6, 7, 8, 9

        # Limit
        limited = memory.get_patterns(machine_id="m1", limit=3)
        assert len(limited) == 3

    def test_get_normal_patterns(self, memory):
        """Test convenience method for normal patterns."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        for i in range(5):
            memory.store_pattern(PatternRecord(
                machine_id="m1",
                timestamp=datetime.now(),
                feature_vector=np.array([i]),
                is_anomaly=False,
            ))

        for i in range(3):
            memory.store_pattern(PatternRecord(
                machine_id="m1",
                timestamp=datetime.now(),
                feature_vector=np.array([i]),
                is_anomaly=True,
            ))

        normals = memory.get_normal_patterns("m1")
        assert len(normals) == 5

    def test_get_anomaly_patterns(self, memory):
        """Test convenience method for anomaly patterns."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        for i in range(5):
            memory.store_pattern(PatternRecord(
                machine_id="m1",
                timestamp=datetime.now(),
                feature_vector=np.array([i]),
                is_anomaly=False,
            ))

        for i in range(3):
            memory.store_pattern(PatternRecord(
                machine_id="m1",
                timestamp=datetime.now(),
                feature_vector=np.array([i]),
                is_anomaly=True,
            ))

        anomalies = memory.get_anomaly_patterns("m1")
        assert len(anomalies) == 3

    def test_update_pattern_label(self, memory):
        """Test label updating."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        pattern_id = memory.store_pattern(PatternRecord(
            machine_id="m1",
            timestamp=datetime.now(),
            feature_vector=np.array([1, 2, 3]),
            label="unknown",
        ))

        memory.update_pattern_label(pattern_id, "confirmed_fault")

        retrieved = memory.get_pattern(pattern_id)
        assert retrieved.label == "confirmed_fault"

    def test_delete_pattern(self, memory):
        """Test single pattern deletion."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        pattern_id = memory.store_pattern(PatternRecord(
            machine_id="m1",
            timestamp=datetime.now(),
            feature_vector=np.array([1, 2, 3]),
        ))

        result = memory.delete_pattern(pattern_id)
        assert result is True

        retrieved = memory.get_pattern(pattern_id)
        assert retrieved is None

    def test_delete_patterns_bulk(self, memory):
        """Test bulk pattern deletion."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        # Store patterns
        for i in range(10):
            memory.store_pattern(PatternRecord(
                machine_id="m1",
                timestamp=datetime.now(),
                feature_vector=np.array([i]),
                is_anomaly=(i % 2 == 0),
            ))

        # Delete anomalies only
        deleted = memory.delete_patterns(machine_id="m1", is_anomaly=True)
        assert deleted == 5

        # Check remaining
        remaining = memory.get_patterns(machine_id="m1")
        assert len(remaining) == 5

    def test_get_pattern_count(self, memory):
        """Test pattern counting."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        for i in range(10):
            memory.store_pattern(PatternRecord(
                machine_id="m1",
                timestamp=datetime.now(),
                feature_vector=np.array([i]),
                is_anomaly=(i % 3 == 0),
            ))

        total = memory.get_pattern_count(machine_id="m1")
        assert total == 10

        anomalies = memory.get_pattern_count(machine_id="m1", is_anomaly=True)
        assert anomalies == 4

    def test_get_label_counts(self, memory):
        """Test label counting."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        labels = ["normal", "normal", "normal", "fault", "fault", None]
        for label in labels:
            memory.store_pattern(PatternRecord(
                machine_id="m1",
                timestamp=datetime.now(),
                feature_vector=np.array([1]),
                label=label,
            ))

        counts = memory.get_label_counts("m1")
        assert counts["normal"] == 3
        assert counts["fault"] == 2
        assert counts.get("unlabeled", 0) == 1

    def test_get_statistics(self, memory):
        """Test statistics collection."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        for i in range(10):
            memory.store_pattern(PatternRecord(
                machine_id="m1",
                timestamp=datetime.now(),
                feature_vector=np.array([i]),
                is_anomaly=(i >= 8),
                anomaly_score=i / 10,
            ))

        stats = memory.get_statistics("m1")

        assert stats["total_patterns"] == 10
        assert stats["normal_patterns"] == 8
        assert stats["anomaly_patterns"] == 2
        assert abs(stats["anomaly_rate"] - 0.2) < 0.01
        assert stats["score_min"] == 0.0
        assert stats["score_max"] == 0.9

    def test_get_feature_vectors(self, memory):
        """Test feature vector extraction."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        for i in range(5):
            memory.store_pattern(PatternRecord(
                machine_id="m1",
                timestamp=datetime.now(),
                feature_vector=np.array([i, i*2, i*3], dtype=np.float64),
                is_anomaly=False,
            ))

        vectors = memory.get_feature_vectors("m1")
        assert vectors.shape == (5, 3)
        # Results are ordered by timestamp DESC, so check that all expected values are present
        assert set(vectors[:, 0]) == {0, 1, 2, 3, 4}
        assert set(vectors[:, 2]) == {0, 3, 6, 9, 12}

    def test_export_to_csv(self, memory, tmp_path):
        """Test CSV export."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        for i in range(5):
            memory.store_pattern(PatternRecord(
                machine_id="m1",
                timestamp=datetime.now(),
                feature_vector=np.array([i, i*2], dtype=np.float64),
                is_anomaly=False,
                label="normal",
            ))

        csv_path = tmp_path / "export.csv"
        count = memory.export_to_csv(str(csv_path), machine_id="m1")

        assert count == 5
        assert csv_path.exists()

        # Read and verify
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 5
            assert "feature_0" in rows[0]

    def test_nonexistent_machine(self, memory):
        """Test behavior with nonexistent machine."""
        result = memory.get_machine("nonexistent")
        assert result is None

        patterns = memory.get_patterns(machine_id="nonexistent")
        assert len(patterns) == 0

    def test_feature_vector_dtypes(self, memory):
        """Test various feature vector data types."""
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        # Test different dtypes
        dtypes = [np.float32, np.float64, np.int32, np.int64]

        for i, dtype in enumerate(dtypes):
            pattern = PatternRecord(
                machine_id="m1",
                timestamp=datetime.now(),
                feature_vector=np.array([1, 2, 3], dtype=dtype),
            )
            pattern_id = memory.store_pattern(pattern)

            retrieved = memory.get_pattern(pattern_id)
            assert np.allclose(retrieved.feature_vector, [1, 2, 3])


class TestMemoryBankPersistence:
    """Tests for persistent (file-based) memory bank."""

    def test_file_persistence(self, tmp_path):
        """Test that data persists to file."""
        db_path = tmp_path / "test.db"

        # Create and populate
        memory1 = MemoryBank(str(db_path))
        machine = MachineRecord(id="m1", name="M1")
        memory1.register_machine(machine)
        memory1.store_pattern(PatternRecord(
            machine_id="m1",
            timestamp=datetime.now(),
            feature_vector=np.array([1, 2, 3]),
        ))

        # Close and reopen
        memory1.close()
        memory2 = MemoryBank(str(db_path))

        # Data should still be there
        retrieved = memory2.get_machine("m1")
        assert retrieved is not None

        patterns = memory2.get_patterns(machine_id="m1")
        assert len(patterns) == 1

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created."""
        db_path = tmp_path / "subdir" / "deep" / "test.db"
        memory = MemoryBank(str(db_path))

        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        assert db_path.exists()


class TestConcurrency:
    """Tests for concurrent access patterns."""

    def test_multiple_operations(self):
        """Test multiple rapid operations."""
        memory = MemoryBank(":memory:")
        machine = MachineRecord(id="m1", name="M1")
        memory.register_machine(machine)

        # Rapid insert/query operations
        for i in range(100):
            memory.store_pattern(PatternRecord(
                machine_id="m1",
                timestamp=datetime.now(),
                feature_vector=np.array([i]),
            ))

            if i % 10 == 0:
                # Query while inserting
                count = memory.get_pattern_count(machine_id="m1")
                assert count > 0

        final_count = memory.get_pattern_count(machine_id="m1")
        assert final_count == 100
