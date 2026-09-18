"""
run.py — Entry point for the AI Crop Advisor Flask application.

Usage from project root:
    python AI_Crop_Advisor/run.py

Or from inside AI_Crop_Advisor/:
    python run.py
"""

import os
import sys

# Ensure the AI_Crop_Advisor directory is on sys.path so all imports work.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from app.app import create_app

if __name__ == "__main__":
    app = create_app()
    print("\n" + "=" * 60)
    print("  AI Crop Advisor - Flask Server")
    print("  URL: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
