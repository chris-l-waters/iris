"""Tests for hardware detection functionality."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware import get_ram_gb, detect_gpu, recommend_model, get_hardware_info


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
    assert recommend_model(2) == "llama3.2:1b-instruct-q4_K_M"  # Low RAM
    assert recommend_model(4) == "llama3.2:3b-instruct-q4_K_M"  # Medium RAM
    assert recommend_model(8) == "mistral:7b-instruct-q4_K_M"   # High RAM
    assert recommend_model(16) == "mistral:7b-instruct-q4_K_M"  # Very high RAM

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
