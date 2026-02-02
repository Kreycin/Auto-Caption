#!/bin/bash
cd "$(dirname "$0")"
# Kill any existing python instance on port 5001 just in case
lsof -ti:5001 | xargs kill -9 2>/dev/null
python3 webapp.py
