# YouTube Video Assistant (RAG) 🎥🤖

A powerful Retrieval-Augmented Generation (RAG) system that processes YouTube videos and answers questions about their content using AI.

## ✨ Features

- **YouTube Video Processing**: Extract transcripts using multiple fallback methods
- **AI-Powered Q&A**: Ask questions about video content and get intelligent answers
- **Smart Summarization**: Generate concise video summaries
- **Transcript Management**: Download cleaned and processed transcripts
- **Grammar Correction**: Fix user query grammar automatically
- **Vector Search**: Fast similarity search using FAISS/Pinecone
- **Session Management**: Maintain conversation context per video

## 🏗️ Architecture

- **Backend**: FastAPI with async support
- **AI Models**: Azure OpenAI (GPT-4, Embeddings)
- **Vector Database**: Pinecone (with FAISS fallback)
- **Transcript Processing**: YouTube API + Whisper fallback
- **Caching**: In-memory video and session cache

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Azure OpenAI account
- Pinecone account (optional - has FAISS fallback)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Ask_Tube
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Run the application**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 🔧 Environment Variables

Create a `.env` file with the following variables:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Pinecone (optional)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=gcp-starter
PINECONE_INDEX=youtube-rag

# Optional APIs
TAVILY_API_KEY=your_tavily_key
WEATHERSTACK_API_KEY=your_weather_key
GOOGLE_API_KEY=your_google_key
```

## 📚 API Endpoints

### Core Endpoints

- **`POST /index`** - Process and index YouTube videos
- **`POST /query`** - Ask questions about video content
- **`POST /summarize`** - Generate video summaries
- **`GET /transcript`** - Download cleaned transcripts
- **`POST /check-grammar`** - Fix query grammar
- **`GET /status`** - Check video processing status
- **`GET /health`** - Health check endpoint

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🎯 Usage Examples

### Index a YouTube Video
```bash
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

### Ask a Question
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "query": "What are the main points discussed in this video?"
  }'
```

### Get Summary
```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "max_words": 100
  }'
```

## 🚀 Deployment

### GitHub Deployment

1. **Initialize Git repository**
```bash
git init
git add .
git commit -m "Initial commit: YouTube RAG Assistant"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. **Set up GitHub Actions** (optional)
- The `.github/workflows/deploy.yml` file will automatically deploy to Render

### Render Deployment

1. **Connect your GitHub repository to Render**
2. **Create a new Web Service**
3. **Configure build settings**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Set environment variables** in Render dashboard
5. **Deploy!**

## 🔍 How It Works

1. **Video Input**: User provides YouTube URL
2. **Transcript Extraction**: Multiple fallback methods:
   - YouTube Transcript API (primary)
   - Whisper speech-to-text (fallback)
3. **Chunking**: Split transcript into searchable pieces
4. **Embedding**: Convert text to vectors using Azure OpenAI
5. **Storage**: Store vectors in Pinecone/FAISS
6. **Query Processing**: 
   - Search for relevant chunks
   - Generate AI answers using retrieved context
   - Maintain conversation history

## 🛠️ Development

### Project Structure
```
Ask_Tube/
├── app/
│   ├── services.py      # Core business logic
│   ├── llm.py          # AI model integration
│   └── utils.py        # Utility functions
├── main.py             # FastAPI application
├── requirements.txt     # Python dependencies
├── .env               # Environment variables
├── .gitignore         # Git ignore rules
├── README.md          # This file
└── Procfile           # Render deployment config
```

### Adding New Features

1. **Extend services.py** for new functionality
2. **Add new endpoints** in main.py
3. **Update requirements.txt** for new dependencies
4. **Test locally** before deploying

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Azure OpenAI** for AI capabilities
- **Pinecone** for vector database
- **LangChain** for RAG framework
- **FastAPI** for the web framework
- **Whisper** for speech-to-text fallback

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-username/Ask_Tube/issues) page
2. Create a new issue with detailed information
3. Include error logs and reproduction steps

---

**Made with ❤️ for the AI community**
