#!/bin/bash
# Render deployment setup script
# This runs during the build phase

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Setup complete ==="

