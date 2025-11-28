# SemanticLibrary

SemanticLibrary is a lightweight and easy-to-use library database manager for Windows, enabling semantic search and other AI-powered features for your book collection.

## Features

### Webscraper/Downloader
Input a URL from a supported website and SemanticLibrary will download the book for you and convert it to text if necessary.

### Database Management
SemanticLibrary manages your library by storing book content, identification info, and metadata in a SQLite database.

### Embedding & Vector Search
SemanticLibrary features an embedding pipeline that:
- Separates book contents into overlapping chunks stored in the database
- Embeds those chunks using AI models
- Stores the vectors in a FAISS index for fast similarity search

This enables true semantic search: instead of needing exact keyword matches, you can search by meaning or concept. SemanticLibrary will find the most relevant passages based on semantic similarity, not just word matching.

## How It Works

1. **Download** - Fetch books from supported sources
2. **Process** - Extract and chunk text content
3. **Embed** - Generate semantic vectors for each chunk
4. **Search** - Find relevant passages using natural language queries

## Technologies

- **C# WinForms** - Desktop application
- **Python** - AI/ML backend (sentence-transformers, FAISS)
- **SQLite** - Local database
- **FAISS** - Vector similarity search

## Setup

1. Download the latest release
2. Create a new virtual environment using any Python venv manager:
```bash
   conda create -n SemanticLibrary python=3.10
   conda activate SemanticLibrary
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
   *(ROCm users: uncomment the necessary portions in requirements.txt)*
4. Run `SemanticLibrary.exe`
5. Open **Settings** and configure:
   - Location of your `python.exe` for your venv
   - Location to save the database and FAISS index
   - Click **Apply**
6. Open the **Download** window and add books:
   - **Option 1:** Manually paste the author, URL, and title
   - **Option 2:** Click "Read from file" and import from `.json` (see `template.json`)
7. Click **Embed** to process and index your books
8. Start searching! Type your queries in the search box and click **Search**

## Supported Download Sites

- **Internet Archive**
- **Project Gutenberg**
- **Google Books** *(Note: Google Books often fails due to CAPTCHA. Manual import coming soon.)*

## Requirements

- Windows 10/11
- Python 3.8+
- ~2GB disk space for dependencies
- GPU optional (for faster embedding)