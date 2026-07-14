# Chat With YouTube

This Streamlit app lets a user paste a YouTube URL, load the transcript, and ask questions about the video.

## Files

```text
main.py
requirements.txt
.env.example
.gitignore
README.md
```

## API Key Needed

Yes, you need an OpenAI API key.

Locally, create a file named `.env`:

```text
OPENAI_API_KEY=your_real_openai_api_key_here
```

Do not upload `.env` to GitHub.

For Streamlit Cloud, add this in app secrets:

```toml
OPENAI_API_KEY = "your_real_openai_api_key_here"
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run main.py
```

## How It Works

1. User enters a YouTube URL.
2. App extracts the video ID.
3. App fetches the YouTube transcript.
4. App splits the transcript into chunks.
5. App creates embeddings for each chunk.
6. User asks a question.
7. App finds the most relevant transcript chunks.
8. OpenAI answers using only those chunks.

## Upload To GitHub

Upload these files:

```text
main.py
requirements.txt
.env.example
.gitignore
README.md
```

Do not upload:

```text
.env
```

## Notes

Some YouTube videos do not have transcripts. If transcripts are disabled or missing, the app cannot answer questions about that video.
