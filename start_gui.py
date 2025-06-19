#!/usr/bin/env python3
"""Startup script for IRIS web interface."""

import subprocess
import sys
import os


def main():
    """Start the IRIS web interface."""
    print("Starting IRIS Web Interface...")

    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    # Check if virtual environment exists
    venv_python = os.path.join(project_dir, "venv", "bin", "python")
    if not os.path.exists(venv_python):
        print("Error: Virtual environment not found.")
        print(
            "Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        )
        sys.exit(1)

    # Start the Flask app
    try:
        subprocess.run([venv_python, "gui/app.py"], check=True)
    except KeyboardInterrupt:
        print("\nShutting down IRIS...")
    except FileNotFoundError:
        print("Error: gui/app.py not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
