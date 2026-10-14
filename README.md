# 🔍 TruthLens AI
**Verify Before You Amplify.**

## What it does
TruthLens AI checks a social media post, headline or claim. It extracts the main factual claim, searches existing
fact-checks through Google's Fact Check Tools API, and shows one of three results with an explanation, evidence and
links:

- ✅ **Likely Reliable**
- ⚠️ **Needs Verification**
- 🚫 **Potential Fake/Misinformation**

## The problem it solves
Misinformation spreads faster than corrections. People need a quick way to see whether trusted fact-checkers have
already reviewed a claim, and they need to see *why* something is flagged, not just a label.

## Technologies
Python, Streamlit, Requests, python-dotenv, Google Fact Check Tools API, PubMed E-utilities (optional context),
Pillow + pytesseract (optional OCR).

## How the verification workflow works
1. **Validate** the input (not empty, 10 to 2000 characters).
2. **Extract claims**: the text is split into sentences and ranked by simple rules (numbers, factual verbs, names;
   questions and personal opinions rank lower). The user can pick a different claim.
3. **Search** Google Fact Check Tools API (live) with the claim.
4. **Filter**: only fact-checks whose reviewed claim shares at least 50% of the claim's keywords are used.
5. **Assess** using publishers' ratings:
   - Relevant fact-check(s) rate it false/misleading, none rate it true → **Potential Misinformation**
   - Relevant fact-check(s) rate it true, none rate it false → **Likely Reliable**
   - Conflicting ratings, mixed ratings, no relevant fact-check, or a failed lookup → **Needs Verification**
6. **Explain**: show the evidence, links, review dates, the sources actually queried, and limitations.

No fact-check found never means "false". It always gives "Needs Verification".

## Getting the Google Fact Check API key (free)
1. Go to https://console.cloud.google.com and sign in.
2. Create a new project.
3. Open **APIs & Services → Library**, search **Fact Check Tools API** and click **Enable**.
4. Open **APIs & Services → Credentials → Create credentials → API key**.
5. Copy the key (and optionally restrict it to the Fact Check Tools API).

Quotas may apply and API availability can change.

## Installation (Windows)
1. Install Python 3.10+ from https://www.python.org/downloads/ (tick **"Add Python to PATH"**).
2. Open **Command Prompt** in the project folder:
```
   cd path\to\truthlens-ai
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
```
   *(PowerShell blocked activation? Run `Set-ExecutionPolicy -Scope Process Bypass` first, or use Command Prompt.)*
3. Create your key file:
```
   copy .env.example .env
   notepad .env
```
   Replace `your_api_key_here` with your real key, save, and close.

## Start the app
```
streamlit run app.py
```
The browser opens at http://localhost:8501. If the key is missing the app still starts and shows a clear warning.

## Optional: enable OCR (image upload)
1. Install Tesseract for Windows (the UB Mannheim build is commonly used): https://github.com/UB-Mannheim/tesseract/wiki
2. Keep the default install folder (`C:\Program Files\Tesseract-OCR`) or set `TESSERACT_CMD` in `.env`.
3. Restart the app. OCR only reads text. It does **not** detect edited images or find their original source.

## How to test
- **No key test:** start without a key. You should see the "Live verification is unavailable" warning, and Verify should return Needs Verification with no fake evidence.
- **Live tests:** click a sample input (for example "5G towers spread COVID-19.") and press **Verify Claim**. Whatever appears comes from the live API.
- **Opinion test:** use the opinion sample. It should not be called misinformation.
- **Multi-claim test:** paste two sentences and use the claim selector.
- **Error test:** put a wrong key in `.env`. You should see a clear API-key message.

## Current limitations
- Claim extraction is rule-based, not AI-based.
- Fact-check matching uses keyword overlap only.
- Fact-check coverage is incomplete, and non-English or very new claims often have no results.
- Government and primary-source websites are not searched.
- PubMed papers are context only and are not analysed.
- Image analysis is OCR only (no manipulation or origin detection).
- Works only on text the user submits. It does not monitor social media platforms.

## Possible future improvements
- Optional AI model for claim extraction and summarising evidence (with citations required)
- Searching official sources (WHO, CDC, government statistics) through their APIs
- Semantic matching with embeddings
- Reverse-image search and manipulation detection
- Multi-language support and browser extension
- Saving history in a database

## Security
Your API key lives only in `.env`, which is listed in `.gitignore`. The key is never shown in the app.