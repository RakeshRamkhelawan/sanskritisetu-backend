#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unicode-safe logging configuration
Implements logging best practices from UNICODE_BEST_PRACTICES.md
"""

import logging
import sys
import os

def setup_unicode_logging(level=logging.INFO):
    """Configure logging with Unicode support"""
    
    # Force UTF-8 encoding for stdout/stderr on Windows
    if sys.platform == "win32":
        import codecs
        # Only reconfigure if not already UTF-8
        if not hasattr(sys.stdout, 'encoding') or sys.stdout.encoding.lower() != 'utf-8':
            try:
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
            except (AttributeError, OSError):
                # Fallback for environments where detach() is not available
                pass
    
    # Create formatter with safe format
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create handler with explicit encoding
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=[handler],
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger()
