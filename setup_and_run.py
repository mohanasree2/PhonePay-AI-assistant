#!/usr/bin/env python3
"""
============================================================
SETUP & RUN SCRIPT
============================================================
File: setup_and_run.py

PURPOSE:
  One-click setup script that:
  1. Checks all dependencies are installed
  2. Creates required directories
  3. Validates environment variables
  4. Runs the ingestion pipeline to build the vector store
  5. Launches the Streamlit app

RUN:
  python setup_and_run.py
============================================================
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    print("""
╔══════════════════════════════════════════════════╗
║         💜 PhonePay RAG AI Assistant             ║
║         Setup & Launch Script                    ║
║         Powered by Gemini + LangChain + Chroma   ║
╚══════════════════════════════════════════════════╝
""")


def check_python_version():
    """Ensure Python 3.9+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9 or higher is required.")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} — OK")


def install_requirements():
    """Install all Python packages from requirements.txt"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"
        ])
        print("✅ All packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Package installation failed: {e}")
        print("   Try manually: pip install -r requirements.txt")
        sys.exit(1)


def create_directories():
    """Create required project directories."""
    dirs = ["logs", "chroma_db", "data", "src"]
    print("\n📁 Creating directories...")
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directories ready: logs/, chroma_db/, data/, src/")


def check_env_file():
    """Check if .env file exists with API key."""
    print("\n🔑 Checking environment configuration...")
    
    if not Path(".env").exists():
        if Path(".env.example").exists():
            print("⚠️  .env file not found!")
            print("   Please do the following:")
            print("   1. Copy .env.example to .env")
            print("      Command: cp .env.example .env  (Linux/Mac)")
            print("               copy .env.example .env  (Windows)")
            print("   2. Open .env and add your Google API key:")
            print("      GOOGLE_API_KEY=your_actual_key_here")
            print("   3. Get your key from: https://aistudio.google.com/app/apikey")
            
            create_it = input("\n   Create .env from template now? (y/n): ").strip().lower()
            if create_it == "y":
                import shutil
                shutil.copy(".env.example", ".env")
                print("   ✅ .env created! Please edit it and add your GOOGLE_API_KEY")
                print("      Then re-run this script.")
                sys.exit(0)
        else:
            print("❌ Neither .env nor .env.example found. Please check project files.")
            sys.exit(1)
    
    # Check if API key is set
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your_google_gemini_api_key_here":
        print("❌ GOOGLE_API_KEY is not set or still has placeholder value.")
        print("   Edit .env and set: GOOGLE_API_KEY=your_actual_key")
        print("   Get key from: https://aistudio.google.com/app/apikey")
        sys.exit(1)
    
    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")


def check_data_files():
    """Verify all required data files exist."""
    print("\n📄 Checking data files...")
    
    # UPDATED: Replaced 'user_bills_data.json' with your custom file 'data.json'
    required_files = [
        "data/phonepay_knowledge_base.txt",
        "data/faq_data.txt",
        "data/data.json"
    ]
    
    all_ok = True
    for f in required_files:
        if Path(f).exists():
            size_kb = Path(f).stat().st_size / 1024
            print(f"   ✅ {f} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ {f} — NOT FOUND")
            all_ok = False
    
    if not all_ok:
        print("\n❌ Some data files are missing. Cannot proceed.")
        sys.exit(1)


def run_ingestion():
    """Build the vector store from data files."""
    print("\n🔄 Building vector knowledge base...")
    
    chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    # Check if already built
    if Path(chroma_path).exists() and any(Path(chroma_path).iterdir()):
        print("   ℹ️  ChromaDB already exists.")
        rebuild = input("   Rebuild vector store? (y/n, default=n): ").strip().lower()
        if rebuild != "y":
            print("   ✅ Using existing vector store")
            return
    
    # Add src to path
    sys.path.insert(0, str(Path.cwd()))
    
    try:
        from src.ingest import run_ingestion_pipeline
        run_ingestion_pipeline()
        print("✅ Vector knowledge base built successfully!")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        print("   Check logs/ingest.log for details")
        sys.exit(1)


def launch_app():
    """Launch the Streamlit application."""
    print("\n🚀 Launching PhonePay AI Assistant...")
    print("   URL: http://localhost:8501")
    print("   Press Ctrl+C to stop the server\n")
    
    os.system("streamlit run app.py --server.port 8501 --browser.gatherUsageStats false")


def main():
    print_banner()
    
    check_python_version()
    create_directories()
    install_requirements()
    check_env_file()
    check_data_files()
    run_ingestion()
    launch_app()


if __name__ == "__main__":
    main()