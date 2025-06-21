"""Tests for hardware detection functionality."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware import detect_gpu, get_hardware_info, get_ram_gb, recommend_model


def test_get_ram_gb():
    """Test RAM detection."""
    ram = get_ram_gb()
    assert isinstance(ram, float)
    assert ram > 0
    assert ram < 1000  # Reasonable upper bound


def test_detect_gpu():
    """Test GPU detection."""
    has_gpu = detect_gpu()
    assert isinstance(has_gpu, bool)


def test_recommend_model():
    """Test model recommendation logic."""
    # Test with specific RAM values - RAM-based recommendations
    assert (
        recommend_model(2) == "tinyllama:1.1b-chat-v1-q4_K_M"
    )  # Tier 0: Minimum spec (2-4GB)
    assert (
        recommend_model(4) == "llama3.2:1b-instruct-q4_K_M"
    )  # Tier 1: Low spec (4-6GB)
    assert (
        recommend_model(6) == "llama3.2:3b-instruct-q4_K_M"
    )  # Tier 2: Standard spec (6-8GB)
    assert (
        recommend_model(8) == "gemma2:9b-instruct-q4_K_M"
    )  # Tier 3: High spec (8-12GB)
    assert recommend_model(16) == "phi4-mini:latest"  # Tier 4: Premium spec (12GB+)

    # Test with no RAM specified (should use system RAM)
    model = recommend_model()
    assert isinstance(model, str)


def test_get_hardware_info():
    """Test comprehensive hardware info."""
    info = get_hardware_info()

    # Check all expected keys are present
    expected_keys = {
        "ram_gb",
        "has_gpu",
        "cpu_cores",
        "recommended_model",
        "recommended_embedding",
    }
    assert set(info.keys()) == expected_keys

    # Check types
    assert isinstance(info["ram_gb"], float)
    assert isinstance(info["has_gpu"], bool)
    assert isinstance(info["cpu_cores"], int)
    assert isinstance(info["recommended_model"], str)

    # Check reasonable values
    assert info["ram_gb"] > 0
    assert info["cpu_cores"] > 0
    assert isinstance(info["recommended_model"], str)
    assert isinstance(info["recommended_embedding"], str)
