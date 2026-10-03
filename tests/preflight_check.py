#!/usr/bin/env python3
"""Pre-flight validation script for P4Harness environment.

Verifies:
1. Python dependencies (pytest, pytest-timeout, langgraph, langchain_core, prometheus_client)
2. p4c compiler container / native execution
3. Ollama connectivity & model availability
4. Pytest subprocess execution with --timeout flag
"""
import sys
import subprocess
import shutil

def test_imports():
    print("[1/4] Checking python dependencies...")
    required = [
        "pytest", "pytest_timeout", "langgraph", 
        "langchain_core", "langchain_ollama", 
        "prometheus_client", "rich", "typer"
    ]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise RuntimeError(f"Missing required python modules: {missing}")
    print("  -> All Python dependencies available.")

def test_p4c():
    print("[2/4] Checking P4 compiler execution...")
    # Native
    if shutil.which("p4c-bm2-ss"):
        res = subprocess.run(["p4c-bm2-ss", "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            print("  -> Native p4c-bm2-ss detected.")
            return
    # Apptainer
    if shutil.which("apptainer"):
        res = subprocess.run(
            ["apptainer", "exec", "docker://p4lang/p4c:latest", "p4c-bm2-ss", "--version"],
            capture_output=True, text=True
        )
        if res.returncode == 0:
            print("  -> Apptainer p4c-bm2-ss verified.")
            return
    print("  -> WARNING: No functional p4c compiler detected.")

def test_pytest_timeout():
    print("[3/4] Testing pytest subprocess execution with --timeout...")
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "--help"],
        capture_output=True, text=True
    )
    if "--timeout=" not in res.stdout:
        raise RuntimeError("pytest-timeout is not functioning with active python interpreter!")
    print("  -> pytest --timeout flag recognized successfully.")

def test_ollama():
    print("[4/4] Testing Ollama connectivity...")
    import urllib.request
    import json
    try:
        req = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        data = json.loads(req.read().decode())
        models = [m.get("name") for m in data.get("models", [])]
        print(f"  -> Ollama is live. Loaded models: {models}")
    except Exception as e:
        print(f"  -> NOTICE: Ollama not running on localhost:11434 ({e}). (Expected before server launch)")

if __name__ == "__main__":
    print("=== P4Harness Pre-Flight Self-Test ===")
    test_imports()
    test_p4c()
    test_pytest_timeout()
    test_ollama()
    print("=== Pre-Flight Verification PASSED ===")
