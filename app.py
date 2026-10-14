"""
TruthLens AI - "Verify Before You Amplify."

A Streamlit app that helps people check social-media claims.

How it works (short version):
1. The user pastes text (or uploads an image, which is read with OCR).
2. We pick the main factual claim using simple, transparent rules.
3. We search Google's Fact Check Tools API for existing fact-checks.
4. We keep only fact-checks that closely match the claim.
5. We assess the result from the PUBLISHERS' ratings (not from AI opinion).
6. We show the result with evidence, links, and limitations.

Nothing in this app is mock data. If a lookup fails, the app says so.
"""

import html
import os
import re
from datetime import datetime
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv

# OCR is optional. If these libraries are missing, the app still works.
try:
    from PIL import Image
    import pytesseract

    OCR_IMPORTS_OK = True
except ImportError:
    OCR_IMPORTS_OK = False

# Read variables from the .env file (if it exists) into the environment.
load_dotenv()

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

REQUEST_TIMEOUT = 10        # seconds; every web request has a timeout
MIN_CHARS = 10              # shortest text we accept
MAX_CHARS = 2000            # longest text we accept
RELEVANCE_THRESHOLD = 0.5   # how closely a fact-check must match our claim

# Windows users: if Tesseract is installed in the default folder, use it.
if OCR_IMPORTS_OK:
    _custom_cmd = os.getenv("TESSERACT_CMD", "").strip()
    _default_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if _custom_cmd:
        pytesseract.pytesseract.tesseract_cmd = _custom_cmd
    elif os.path.exists(_default_cmd):
        pytesseract.pytesseract.tesseract_cmd = _default_cmd

# Labels, icons and CSS classes for the three result categories.
CATEGORIES = {
    "green": {"label": "Likely Reliable", "icon": "✅", "css": "tl-green"},
    "yellow": {"label": "Needs Verification", "icon": "⚠️", "css": "tl-yellow"},
    "red": {"label": "Potential Fake/Misinformation", "icon": "🚫", "css": "tl-red"},
}

# Sample INPUTS for testing. These are NOT results. Results always come from
# live lookups.
SAMPLE_CLAIMS = [
    "5G towers spread COVID-19.",
    "Vaccines cause autism.",
    "The Great Wall of China is visible from space with the naked eye.",
    "I think this new phone is the worst thing ever released!!!",
]

# Common words ignored when comparing texts.
STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "that", "this", "with", "from",
    "has", "have", "had", "not", "but", "you", "your", "they", "their", "its",
    "his", "her", "she", "him", "who", "what", "when", "where", "why", "how",
    "can", "will", "would", "could", "should", "been", "being", "than", "then",
    "there", "these", "those", "into", "about", "over", "also", "just", "all",
    "any", "out", "our", "very", "more", "most", "some", "such", "does", "did",
    "is", "it", "in", "on", "of", "to", "a", "an", "as", "at", "by", "be", "or",
    "if", "so", "we", "he", "me", "my", "do", "no", "up",
}

# Words that often appear in checkable factual statements.
FACT_CUES = {
    "cause", "causes", "caused", "cure", "cures", "cured", "kill", "kills",
    "prove", "proves", "proved", "announce", "announced", "ban", "banned",
    "confirm", "confirmed", "found", "died", "dead", "increase", "increased",
    "decrease", "rise", "rose", "percent", "study", "studies", "according",
    "report", "reported", "won", "lost", "arrest", "arrested", "contain",
    "contains", "linked", "spread", "spreads", "visible", "caught", "released",
}

OPINION_STARTS = (
    "i think", "i feel", "i believe", "in my opinion", "imo", "personally",
    "i love", "i hate",
)


class LookupFailed(Exception):
    """Raised when an external lookup fails. The message is safe to show."""


# ----------------------------------------------------------------------------
# Small helper functions
# ----------------------------------------------------------------------------
def get_api_key() -> str:
    """Read the Google key from the environment. Returns '' if missing."""
    key = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()
    if key.lower() in ("", "your_api_key_here"):
        return ""
    return key


def esc(value) -> str:
    """Escape text so it is safe to put inside HTML."""
    return html.escape(str(value if value is not None else ""))


def is_safe_url(url: str) -> bool:
    """Only allow http/https links (blocks things like javascript: URLs)."""
    try:
        parsed = urlparse(url or "")
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except ValueError:
        return False


def tokenize(text: str) -> list:
    """Lowercase keywords from a text, without stopwords."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def relevance_score(claim: str, other_text: str) -> float:
    """
    What fraction of OUR claim's keywords appear in the other text?
    0.0 = nothing in common, 1.0 = all keywords found.
    """
    mine = set(tokenize(claim))
    theirs = set(tokenize(other_text))
    if not mine:
        return 0.0
    return len(mine & theirs) / len(mine)


def validate_input(text: str):
    """Return an error message if the text is not acceptable, else None."""
    text = (text or "").strip()
    if not text:
        return "Please paste a post, headline, or claim first."
    if len(text) < MIN_CHARS:
        return f"Please enter at least {MIN_CHARS} characters so there is something to check."
    if len(text) > MAX_CHARS:
        return f"Please keep the text under {MAX_CHARS} characters."
    return None


# ----------------------------------------------------------------------------
# Step 1: Claim extraction (simple, transparent rules - no AI needed)
# ----------------------------------------------------------------------------
def extract_claim_candidates(text: str) -> list:
    """
    Split the text into sentences and rank them by how much they look like a
    checkable factual claim. Returns up to 5 candidates, best first.

    We do NOT judge truth here and we do not treat emotional language as
    suspicious. We only look for things like numbers, factual verbs and names.
    """
    text = re.sub(r"https?://\S+", " ", text or "")   # remove links
    text = re.sub(r"@\w+", " ", text)                  # remove @mentions
    text = text.replace("#", "")                       # keep hashtag words

    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    scored = []
    for index, piece in enumerate(pieces):
        sentence = re.sub(r"\s+", " ", piece).strip()
        words = re.findall(r"\w+", sentence)
        if len(words) < 3:
            continue

        score = 0
        lower = sentence.lower()
        if re.search(r"\d", sentence):
            score += 2                                  # numbers / dates
        if "%" in sentence:
            score += 1
        if any(w in FACT_CUES for w in re.findall(r"[a-z]+", lower)):
            score += 2                                  # factual verbs
        # Capitalised words after the first word often mean names/places.
        capitals = [w for w in sentence.split()[1:] if w[:1].isupper()]
        score += min(len(capitals), 3)
        if sentence.endswith("?"):
            score -= 3                                  # questions
        if lower.startswith(OPINION_STARTS):
            score -= 3                                  # personal opinions

        scored.append((score, index, sentence[:300]))

    if not scored:
        cleaned = re.sub(r"\s+", " ", text).strip()
        return [cleaned[:300]] if cleaned else []

    # Highest score first; earlier sentences win ties.
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [sentence for _, _, sentence in scored[:5]]


def looks_like_opinion(claim: str) -> bool:
    """Rough check used only to add a helpful note for the user."""
    lower = claim.lower()
    return lower.startswith(OPINION_STARTS) or claim.strip().endswith("?")


# ----------------------------------------------------------------------------
# Step 2: Google Fact Check Tools API
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def search_fact_checks(query: str, api_key: str) -> list:
    """
    Search existing fact-checks. Results are cached for an hour so the same
    claim does not call the API again. Errors raise LookupFailed (errors are
    NOT cached).
    """
    params = {"query": query, "key": api_key, "pageSize": 10}
    try:
        response = requests.get(FACT_CHECK_URL, params=params, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        raise LookupFailed("The Fact Check API took too long to respond (timeout).")
    except requests.exceptions.ConnectionError:
        raise LookupFailed("Could not connect to the Fact Check API. Check your internet connection.")
    except requests.exceptions.RequestException as error:
        # We print only the error type, never the full message (it may hold the key URL).
        raise LookupFailed(f"Network problem while contacting the Fact Check API ({type(error).__name__}).")

    if response.status_code == 429:
        raise LookupFailed("Fact Check API rate limit or quota reached. Please wait and try again.")
    if response.status_code in (401, 403):
        raise LookupFailed("The Fact Check API rejected the API key. Check that the key is correct and that the Fact Check Tools API is enabled in your Google Cloud project.")
    if response.status_code == 400:
        raise LookupFailed("The Fact Check API could not process this query (HTTP 400).")
    if response.status_code != 200:
        raise LookupFailed(f"The Fact Check API returned an unexpected status (HTTP {response.status_code}).")

    try:
        data = response.json()
    except ValueError:
        raise LookupFailed("The Fact Check API returned a response that could not be read.")

    reviews, seen_urls = [], set()
    for claim in data.get("claims", []):
        for review in claim.get("claimReview", []):
            url = review.get("url", "")
            if not is_safe_url(url) or url in seen_urls:
                continue
            seen_urls.add(url)
            publisher = review.get("publisher", {})
            reviews.append({
                "publisher": publisher.get("name") or publisher.get("site") or "Unknown publisher",
                "claim_text": claim.get("text", ""),
                "claimant": claim.get("claimant", ""),
                "rating": review.get("textualRating", "") or "No rating provided",
                "title": review.get("title", ""),
                "date": (review.get("reviewDate") or "")[:10],   # YYYY-MM-DD
                "url": url,
            })
    return reviews


def classify_rating(rating: str) -> str:
    """
    Turn a publisher's rating text into: 'false', 'true', 'mixed' or 'unknown'.
    This only INTERPRETS the publisher's words; it is not our own opinion.
    Order matters: mixed first, then false, then true ("not true" must not
    count as true).
    """
    r = (rating or "").lower()
    mixed = ["half true", "half-true", "partly", "partially", "mixture", "mixed",
             "some truth", "missing context", "needs context", "lacks context",
             "unproven", "unverified", "unsupported", "outdated", "exaggerat"]
    false = ["false", "fake", "incorrect", "misleading", "pants on fire",
             "pants-on-fire", "fabricated", "hoax", "untrue", "not true",
             "wrong", "no evidence", "scam", "debunk", "baseless",
             "inaccurate", "manipulated"]
    true = ["true", "correct", "accurate", "legit"]

    if any(word in r for word in mixed):
        return "mixed"
    if any(word in r for word in false):
        return "false"
    if any(word in r for word in true):
        return "true"
    return "unknown"


# ----------------------------------------------------------------------------
# Step 3 (optional): PubMed - scientific literature as CONTEXT only
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def search_pubmed(query: str) -> list:
    """
    Search PubMed (US National Library of Medicine, free, no key needed).
    The papers found are shown as background reading. They are NOT used to
    decide the category, because keyword search cannot tell whether a paper
    supports or contradicts a claim.
    """
    try:
        search = requests.get(
            PUBMED_SEARCH_URL,
            params={"db": "pubmed", "term": query, "retmode": "json",
                    "retmax": 5, "sort": "relevance"},
            timeout=REQUEST_TIMEOUT,
        )
        search.raise_for_status()
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        summary = requests.get(
            PUBMED_SUMMARY_URL,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        summary.raise_for_status()
        result = summary.json().get("result", {})
    except requests.exceptions.Timeout:
        raise LookupFailed("PubMed took too long to respond (timeout).")
    except requests.exceptions.RequestException:
        raise LookupFailed("PubMed could not be reached or returned an error.")
    except ValueError:
        raise LookupFailed("PubMed returned a response that could not be read.")

    papers = []
    for pmid in ids:
        item = result.get(pmid)
        if not item:
            continue
        papers.append({
            "title": item.get("title", "Untitled"),
            "journal": item.get("source", ""),
            "date": item.get("pubdate", ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return papers


# ----------------------------------------------------------------------------
# Step 4: Assessment and the full verification workflow
# ----------------------------------------------------------------------------
def assess(relevant: list, google_ok: bool):
    """
    Decide the category from RELEVANT fact-checks only.
    Returns (category, explanation).
    """
    false_ = [fc for fc in relevant if fc["verdict"] == "false"]
    true_ = [fc for fc in relevant if fc["verdict"] == "true"]

    def names(items):
        return ", ".join(sorted({fc["publisher"] for fc in items}))

    if false_ and true_:
        return "yellow", (
            "Sources conflict. Some fact-check publishers rated similar claims as "
            f"false or misleading ({names(false_)}), while others rated them as "
            f"true or accurate ({names(true_)}). TruthLens cannot resolve this "
            "automatically, so please read the fact-checks yourself."
        )
    if false_:
        return "red", (
            f"{len(false_)} relevant fact-check(s) from independent publishers "
            f"({names(false_)}) rated this claim, or a very similar one, as false "
            "or misleading. This category is based on those publishers' ratings, "
            "not on an AI model's opinion."
        )
    if true_:
        return "green", (
            f"{len(true_)} relevant fact-check(s) from independent publishers "
            f"({names(true_)}) rated this claim, or a very similar one, as true or "
            "accurate, and no contradicting fact-check was found. Check that the "
            "reviewed claim really matches the exact wording and date of the post."
        )
    if relevant:
        return "yellow", (
            "Relevant fact-checks exist, but their ratings are mixed, partial, or "
            "unclear (for example 'partly true' or 'missing context'). The claim "
            "needs a careful human reading."
        )
    if google_ok:
        return "yellow", (
            "The fact-check search ran successfully but found no closely matching "
            "fact-check. This does NOT mean the claim is true or false. The system "
            "simply cannot determine it from the sources it checked."
        )
    return "yellow", (
        "The live fact-check lookup could not be completed, so TruthLens has no "
        "evidence to evaluate this claim. No verdict can be given."
    )


def run_verification(raw_text: str, claim: str, use_pubmed: bool) -> dict:
    """Run the whole workflow and return one result dictionary."""
    errors, sources_queried = [], []
    all_checks, google_ok = [], False

    # --- Google Fact Check Tools API -----------------------------------
    api_key = get_api_key()
    if not api_key:
        errors.append(
            "Live fact-check verification is unavailable: GOOGLE_FACT_CHECK_API_KEY "
            "is not set. See the README to add your key. No verification was performed."
        )
    else:
        try:
            all_checks = search_fact_checks(claim[:300], api_key)
            google_ok = True
            sources_queried.append("Google Fact Check Tools API (live search)")
        except LookupFailed as error:
            errors.append(str(error))

    # Keep only fact-checks that closely match the claim.
    relevant, weak = [], []
    for fc in all_checks:
        fc = dict(fc)
        fc["relevance"] = relevance_score(claim, fc["claim_text"] + " " + fc["title"])
        fc["verdict"] = classify_rating(fc["rating"])
        (relevant if fc["relevance"] >= RELEVANCE_THRESHOLD else weak).append(fc)

    # --- PubMed (context only) -----------------------------------------
    papers = []
    if use_pubmed:
        keywords = list(dict.fromkeys(tokenize(claim)))[:6]
        if keywords:
            try:
                papers = search_pubmed(" ".join(keywords))
                sources_queried.append("PubMed / NCBI (live search, context only)")
            except LookupFailed as error:
                errors.append(str(error))

    category, explanation = assess(relevant, google_ok)

    limitations = [
        "Claim extraction uses simple rules, not AI, so it can pick the wrong sentence. Use the claim selector when several claims are found.",
        "Fact-checks are matched to the claim by keyword overlap. A fact-check may be about a different version, place, or date of the claim.",
        "Only Google's Fact Check database" + (" and PubMed were" if use_pubmed else " was")
        + " searched. Official government and scientific organisation websites (for example WHO, CDC, NASA) were NOT searched.",
        "Fact-check coverage is incomplete. Many new or local claims have no fact-check yet.",
    ]
    if looks_like_opinion(claim):
        limitations.append("This text looks like an opinion or a question rather than a checkable fact. Opinions are not misinformation.")
    if papers:
        limitations.append("PubMed papers are background reading only. TruthLens did not read them or decide whether they support the claim.")

    return {
        "time": datetime.now().strftime("%d %b %Y, %H:%M"),
        "submitted": raw_text,
        "claim": claim,
        "category": category,
        "explanation": explanation,
        "supporting": [fc for fc in relevant if fc["verdict"] == "true"],
        "contradicting": [fc for fc in relevant if fc["verdict"] == "false"],
        "other": [fc for fc in relevant if fc["verdict"] in ("mixed", "unknown")],
        "weak": weak,
        "papers": papers,
        "errors": errors,
        "sources_queried": sources_queried,
        "limitations": limitations,
    }


# ----------------------------------------------------------------------------
# Step 5: Interface helpers
# ----------------------------------------------------------------------------
CSS = """
<style>
.block-container {max-width: 900px; padding-top: 1.5rem;}
.tl-hero {text-align:center; padding:1.8rem 1rem; border-radius:18px; color:#fff;
  background:linear-gradient(135deg,#1e3a8a 0%,#0284c7 100%); margin-bottom:1rem;}
.tl-hero-title {font-size:2.6rem; font-weight:800; letter-spacing:-0.5px;}
.tl-hero-tag {font-size:1.2rem; font-weight:500; opacity:.95; margin-top:.2rem;}
.tl-hero-desc {font-size:.98rem; opacity:.9; margin-top:.8rem;}
.tl-status-card {border-radius:14px; padding:1.1rem 1.3rem; margin:1rem 0; border-left:10px solid;}
.tl-status-title {font-size:1.6rem; font-weight:800;}
.tl-status-sub {font-size:.85rem; opacity:.85;}
.tl-green {background:#ecfdf5; border-color:#16a34a; color:#14532d;}
.tl-yellow {background:#fffbeb; border-color:#f59e0b; color:#78350f;}
.tl-red {background:#fef2f2; border-color:#dc2626; color:#7f1d1d;}
.tl-quote {background:#f1f5f9; border-left:4px solid #64748b; padding:.7rem 1rem;
  border-radius:8px; color:#0f172a; white-space:pre-wrap; word-wrap:break-word;}
.tl-fc {background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
  padding:.9rem 1rem; margin:.5rem 0; color:#0f172a; word-wrap:break-word;}
.tl-fc a {color:#1d4ed8; font-weight:600;}
.tl-pill {display:inline-block; background:#e0e7ff; color:#3730a3; border-radius:999px;
  padding:2px 10px; font-size:.8rem; font-weight:700;}
.tl-small {font-size:.82rem; color:#475569;}
@media (max-width: 640px) {
  .tl-hero-title {font-size:1.8rem;}
  .tl-hero-tag {font-size:1rem;}
  .tl-status-title {font-size:1.25rem;}
}
</style>
"""


def factcheck_card(fc: dict) -> str:
    """Build the HTML for one fact-check (everything is escaped)."""
    date = fc["date"] if fc["date"] else "Not provided"
    claimant = f" · Claimant: {esc(fc['claimant'])}" if fc["claimant"] else ""
    title = f"<div class='tl-small'>Article: {esc(fc['title'])}</div>" if fc["title"] else ""
    return (
        "<div class='tl-fc'>"
        f"<b>{esc(fc['publisher'])}</b>"
        f"<div><span class='tl-small'>Publisher's rating:</span> <span class='tl-pill'>{esc(fc['rating'])}</span></div>"
        f"<div style='margin-top:.4rem'><span class='tl-small'>Claim reviewed:</span> {esc(fc['claim_text'])}</div>"
        f"{title}"
        f"<div class='tl-small'>Review date: {esc(date)}{claimant} · Match with your claim: {int(fc['relevance'] * 100)}%</div>"
        f"<div style='margin-top:.3rem'><a href='{esc(fc['url'])}' target='_blank' rel='noopener noreferrer'>Read the full fact-check ↗</a></div>"
        "</div>"
    )


def show_factchecks(title: str, items: list, empty_text: str):
    """Show a titled group of fact-check cards."""
    st.markdown(f"#### {title}")
    if not items:
        st.caption(empty_text)
        return
    for fc in items:
        st.markdown(factcheck_card(fc), unsafe_allow_html=True)


def render_result(res: dict):
    """Display the full explainable result."""
    cat = CATEGORIES[res["category"]]
    st.markdown(
        f"<div class='tl-status-card {cat['css']}'>"
        f"<div class='tl-status-title'>{cat['icon']} {cat['label']}</div>"
        f"<div class='tl-status-sub'>TruthLens AI's own assessment · {esc(res['time'])}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    for message in res["errors"]:
        st.warning(message)

    st.markdown("#### Submitted content")
    st.markdown(f"<div class='tl-quote'>{esc(res['submitted'])}</div>", unsafe_allow_html=True)
    st.markdown("#### Main claim checked")
    st.markdown(f"<div class='tl-quote'>{esc(res['claim'])}</div>", unsafe_allow_html=True)

    st.markdown("#### Why this result?")
    st.write(res["explanation"])
    st.caption("Note: a publisher's rating belongs to that publisher. The category above is TruthLens AI's own assessment built from those ratings.")

    show_factchecks("Contradicting evidence", res["contradicting"],
                    "No relevant fact-check rating this claim as false or misleading was retrieved.")
    show_factchecks("Supporting evidence", res["supporting"],
                    "No relevant fact-check rating this claim as true was retrieved.")
    if res["other"]:
        show_factchecks("Other relevant fact-checks (mixed or unclear ratings)", res["other"], "")

    if res["weak"]:
        with st.expander(f"Low-relevance results returned by the API ({len(res['weak'])}) - not used in the assessment"):
            for fc in res["weak"]:
                st.markdown(factcheck_card(fc), unsafe_allow_html=True)

    if res["papers"]:
        st.markdown("#### Background reading from PubMed (context only)")
        st.caption("These papers matched the claim's keywords. TruthLens did not read them and did not use them to choose the category.")
        for paper in res["papers"]:
            st.markdown(
                f"<div class='tl-fc'><b>{esc(paper['title'])}</b>"
                f"<div class='tl-small'>{esc(paper['journal'])} · {esc(paper['date'])}</div>"
                f"<a href='{esc(paper['url'])}' target='_blank' rel='noopener noreferrer'>Open on PubMed ↗</a></div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Sources actually queried in this check")
    if res["sources_queried"]:
        for source in res["sources_queried"]:
            st.markdown(f"- {source}")
    else:
        st.markdown("- None. No live lookup succeeded in this check.")

    st.markdown("#### Limitations and uncertainty")
    for item in res["limitations"]:
        st.markdown(f"- {item}")


# ----------------------------------------------------------------------------
# Callbacks (run BEFORE the page redraws, so they can safely change widgets)
# ----------------------------------------------------------------------------
def set_sample(text: str):
    st.session_state.input_text = text
    st.session_state.result = None


def clear_all():
    st.session_state.input_text = ""
    st.session_state.result = None
    st.session_state.ocr_msg = None


def run_ocr():
    """Read text from the uploaded image and put it in the text box."""
    uploaded = st.session_state.get("image_upload")
    if uploaded is None:
        st.session_state.ocr_msg = ("warning", "Please choose an image first.")
        return
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        st.session_state.ocr_msg = ("error",
            "The Tesseract program is not installed or was not found. See the README (OCR section) for the Windows installer.")
        return
    try:
        image = Image.open(uploaded).convert("RGB")
        text = pytesseract.image_to_string(image).strip()
    except Exception:
        st.session_state.ocr_msg = ("error", "The image could not be read. Try a clearer PNG or JPG.")
        return

    text = re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", "\n", text))
    if len(text) < MIN_CHARS:
        st.session_state.ocr_msg = ("warning", "Very little text was found in the image. Try a sharper, larger screenshot.")
        return
    st.session_state.input_text = text[:MAX_CHARS]
    st.session_state.result = None
    st.session_state.ocr_msg = ("success",
        "Text extracted. Please read it and correct any OCR mistakes before verifying. "
        "TruthLens only reads text; it does not detect edited or manipulated images.")


# ----------------------------------------------------------------------------
# The page
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="TruthLens AI", page_icon="🔍", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    # Session state keeps data between button clicks.
    for key, default in {"input_text": "", "result": None, "history": [], "ocr_msg": None}.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ---- Hero ----
    st.markdown(
        "<div class='tl-hero'>"
        "<div class='tl-hero-title'>🔍 TruthLens AI</div>"
        "<div class='tl-hero-tag'>Verify Before You Amplify.</div>"
        "<div class='tl-hero-desc'>False claims, misleading headlines and fabricated stories spread "
        "faster than corrections. Paste a post, headline or claim and TruthLens searches existing "
        "fact-checks, then explains what it found, with sources.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if not get_api_key():
        st.warning("⚠️ **Live verification is unavailable.** `GOOGLE_FACT_CHECK_API_KEY` is not set. "
                   "You can still explore the app, but no fact-check search will run. See the README to add a free key.")

    # ---- Samples ----
    st.markdown("**Sample inputs to try** (these are only test inputs, not results; every result comes from a live lookup)")
    for i, sample in enumerate(SAMPLE_CLAIMS):
        st.button(f"Try: {sample[:60]}", key=f"sample_{i}", on_click=set_sample, args=(sample,),
                  use_container_width=True)

    # ---- Input ----
    st.text_area(
        "Paste a social media post, headline, or claim",
        key="input_text", height=170, max_chars=MAX_CHARS,
        placeholder="Example: Scientists announced that drinking lemon water cures cancer...",
    )

    # Optional image upload (OCR)
    with st.expander("📷 Optional: extract text from an image (OCR)"):
        if not OCR_IMPORTS_OK:
            st.info("OCR libraries are not installed. Run `pip install pillow pytesseract` and install "
                    "the Tesseract program (see README). The rest of the app works without it.")
        else:
            st.file_uploader("Upload a screenshot of a post or headline", type=["png", "jpg", "jpeg"],
                             key="image_upload")
            st.button("Extract text from image", on_click=run_ocr)
            st.caption("OCR only reads text. TruthLens does NOT detect edited images or find an image's original source.")
            if st.session_state.ocr_msg:
                level, message = st.session_state.ocr_msg
                getattr(st, level)(message)

    use_pubmed = st.checkbox("Also search PubMed for background scientific papers (context only)", value=True)

    # Claim extraction (updates every time the text changes)
    text = st.session_state.input_text
    candidates = extract_claim_candidates(text) if text.strip() else []
    claim = ""
    if len(candidates) > 1:
        claim = st.selectbox("Several possible claims were found. Choose the one to verify "
                             "(the first is TruthLens's best guess):", candidates)
    elif len(candidates) == 1:
        claim = candidates[0]

    col_verify, col_clear = st.columns([3, 1])
    with col_verify:
        verify_clicked = st.button("🔎 Verify Claim", type="primary", use_container_width=True)
    with col_clear:
        st.button("Clear", on_click=clear_all, use_container_width=True)

    # ---- Run verification ----
    if verify_clicked:
        problem = validate_input(text)
        if problem:
            st.warning(problem)
        elif not claim:
            st.warning("No claim could be found in this text. Please write it as a complete sentence.")
        else:
            with st.spinner("Checking fact-check databases... this can take a few seconds."):
                st.session_state.result = run_verification(text.strip(), claim, use_pubmed)
            res = st.session_state.result
            st.session_state.history.insert(0, {"time": res["time"], "claim": res["claim"], "category": res["category"]})
            st.session_state.history = st.session_state.history[:10]

    if st.session_state.result:
        st.divider()
        render_result(st.session_state.result)

    # ---- History ----
    if st.session_state.history:
        st.divider()
        st.subheader("🕘 Checks in this session")
        for item in st.session_state.history:
            cat = CATEGORIES[item["category"]]
            st.markdown(f"{cat['icon']} **{cat['label']}** · {esc(item['claim'][:90])} <span class='tl-small'>({esc(item['time'])})</span>",
                        unsafe_allow_html=True)

    # ---- Information sections ----
    st.divider()
    st.subheader("How It Works")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1. Input**\n\nPaste text or upload an image (OCR).")
    c2.markdown("**2. Claim**\n\nRules pick the main factual claim; you can choose another.")
    c3.markdown("**3. Search**\n\nGoogle Fact Check API is queried; matches are filtered.")
    c4.markdown("**4. Explain**\n\nA category, evidence, links and limitations are shown.")

    with st.expander("About This Project"):
        st.write("TruthLens AI was built for a college hackathon on *AI-Based Social Media Misinformation Detection*. "
                 "It helps people pause and check a claim before sharing it. It checks only text that "
                 "a user submits. It does not monitor social media platforms automatically.")

    with st.expander("Source transparency"):
        st.markdown(
            "- **Google Fact Check Tools API**: searches fact-checks published by independent organisations. "
            "Ratings shown are the publishers' own.\n"
            "- **PubMed (NCBI)**: optional background papers; not used to decide the category.\n"
            "- **Not searched**: government sites, WHO, CDC and other primary sources. This is a known limitation.\n"
            "- **No AI model** is used to decide results. The category comes from rules applied to fact-check ratings.\n"
            "- Nothing displayed is mock data. If a lookup fails, the app says so."
        )

    st.info("**Disclaimer:** Automated assessments can be wrong. A 'Needs Verification' result means "
            "the system could not decide, not that the claim is true or false. Always read the linked "
            "sources and use your own judgement.")


if __name__ == "__main__":
    main()