#!/usr/bin/env bash
# Print the NAMES of keys in .env.example (never values). Used by the guard-secrets hint.
grep -oE '^[A-Z_]+' .env.example 2>/dev/null || echo "(no .env.example)"
