# Streamlit-only TTS App

Deploy to Streamlit Community Cloud or Hugging Face Spaces (Streamlit). No Flask/CORS.

## Steps
1. Upload this folder to GitHub.
2. On Streamlit Cloud: New app → set main file to `tts_app.py`.
3. Set secret `OPENAI_API_KEY` in the app settings.
4. Deploy.

## Notes
- Use the UI to generate and download MP3. No REST endpoints are exposed.
