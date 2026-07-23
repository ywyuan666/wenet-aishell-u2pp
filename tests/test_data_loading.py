#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for data loading utilities (no model dependencies)."""

import json
import tempfile
from pathlib import Path

import pytest

from run_eval import load_test_items


class TestLoadTestItems:
    """Tests for load_test_items: data.list parsing."""

    @pytest.fixture
    def temp_s0_dir(self):
        """Create a temporary s0 directory with sample data.list."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data" / "test"
            data_dir.mkdir(parents=True)
            data_list = data_dir / "data.list"

            items = [
                {"key": "utt_001", "wav": "/data/wav/001.wav", "txt": "你好"},
                {"key": "utt_002", "wav": "/data/wav/002.wav", "txt": "世界"},
                {"key": "utt_003", "wav": "/data/wav/003.wav", "txt": "测试"},
            ]
            with open(data_list, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            yield str(tmp)

    @pytest.fixture
    def temp_empty_dir(self):
        """Temporary directory without data.list."""
        with tempfile.TemporaryDirectory() as tmp:
            yield str(tmp)

    def test_load_all(self, temp_s0_dir):
        """Loading without subset should return all items."""
        items = load_test_items(temp_s0_dir)
        assert len(items) == 3
        assert items[0]["key"] == "utt_001"
        assert items[2]["txt"] == "测试"

    def test_load_with_subset(self, temp_s0_dir):
        """Loading with subset should return fewer items."""
        items = load_test_items(temp_s0_dir, subset=2)
        assert len(items) == 2

    def test_subset_larger_than_dataset(self, temp_s0_dir):
        """Subset larger than available items returns all."""
        items = load_test_items(temp_s0_dir, subset=10)
        assert len(items) == 3

    def test_empty_data_list(self, temp_empty_dir):
        """Missing data.list should exit with error."""
        with pytest.raises(SystemExit):
            load_test_items(temp_empty_dir)

    def test_item_fields(self, temp_s0_dir):
        """Each item should have key, wav, txt."""
        items = load_test_items(temp_s0_dir)
        for item in items:
            assert "key" in item
            assert "wav" in item
            assert "txt" in item
