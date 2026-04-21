"""Verify the class environment is ready. Run: python verify_setup.py"""
from __future__ import annotations

import os
import sys
from importlib import import_module

REQUIRED = [
    "langgraph",
    "langchain_google_genai",
    "langchain_core",
    "faker",
    "dotenv",
]
OPTIONAL = [
    "langchain_anthropic",
    "langchain_openai",
    "langsmith",
    "anthropic",
]


def check_imports() -> list[str]:
    print("Checking imports...")
    missing = []
    for pkg in REQUIRED:
        try:
            import_module(pkg)
            print(f"  [ok]   {pkg}")
        except ImportError:
            print(f"  [FAIL] {pkg}  (required)")
            missing.append(pkg)
    for pkg in OPTIONAL:
        try:
            import_module(pkg)
            print(f"  [ok]   {pkg}  (optional)")
        except ImportError:
            print(f"  [skip] {pkg}  (optional, skip if not using)")
    return missing


def check_keys() -> bool:
    print("\nChecking API keys...")
    google = os.getenv("GOOGLE_API_KEY")
    anthropic_k = os.getenv("ANTHROPIC_API_KEY")
    openrouter = os.getenv("OPENROUTER_API_KEY")
    langsmith = os.getenv("LANGSMITH_API_KEY")
    print(f"  {'[ok]  ' if google else '[FAIL]'} GOOGLE_API_KEY        (required)")
    print(f"  {'[ok]  ' if anthropic_k else '[skip]'} ANTHROPIC_API_KEY     (optional, Ex 5)")
    print(f"  {'[ok]  ' if openrouter else '[skip]'} OPENROUTER_API_KEY    (optional, swap demos: GLM-5.1 / Qwen3 Coder / DeepSeek R1)")
    print(f"  {'[ok]  ' if langsmith else '[skip]'} LANGSMITH_API_KEY     (optional, Ex 4)")
    return bool(google)


def smoke_test_gemini() -> bool:
    print("\nSmoke test — calling Gemini 2.5 Flash...")
    if not os.getenv("GOOGLE_API_KEY"):
        print("  [skip] no GOOGLE_API_KEY")
        return False
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        r = llm.invoke("Reply in exactly 3 words: env is ready")
        print(f"  [ok] {r.content.strip()}")
        return True
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    missing = check_imports()
    has_google = check_keys()
    if missing:
        print(f"\nInstall missing packages:\n  uv pip install {' '.join(missing)}")
        sys.exit(1)
    if not has_google:
        print("\nSet GOOGLE_API_KEY in .env or your shell before continuing.")
        print("Get a free key at: https://aistudio.google.com/apikey")
        sys.exit(1)
    if not smoke_test_gemini():
        print("\nSmoke test failed. Check your key or network.")
        sys.exit(1)
    print("\nAll set. See you in class.")
