"""Convenience alias entry point for Federated Learning Server."""
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_server import main

if __name__ == "__main__":
    main()
