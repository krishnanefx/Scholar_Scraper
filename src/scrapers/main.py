
import asyncio
import time
import aiohttp
import joblib
from functools import lru_cache
import requests
import mwparserfromhell
import re
import time
import difflib
from transformers import pipeline
import torch
import os
import random
import json
import pandas as pd
import tldextract
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import networkx as nx
from tqdm import tqdm
from collections import Counter, defaultdict, deque
from fuzzywuzzy import process, fuzz
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from multiprocessing import Pool, cpu_count
import threading


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === PATH SETUP ===
def get_project_root():
    """Get the project root directory (Scholar_Scraper)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up from src/scrapers to Scholar_Scraper root
    return os.path.dirname(os.path.dirname(current_dir))

PROJECT_ROOT = get_project_root()

# === CACHING SETUP ===
WIKI_CACHE_PATH = os.path.join(PROJECT_ROOT, "cache", "wiki_lookup_cache.joblib")
FUZZY_CACHE_PATH = os.path.join(PROJECT_ROOT, "cache", "fuzzy_match_cache.joblib")

try:
    wiki_cache = joblib.load(WIKI_CACHE_PATH)
except Exception:
    wiki_cache = {}
try:
    fuzzy_cache = joblib.load(FUZZY_CACHE_PATH)
except Exception:
    fuzzy_cache = {}

# === DEVICE & MODEL SETUP ===
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

print("🔧 Setting up device and model...")
device = get_device()
print(f"📱 Device set to use {device}")

try:
    print("🤖 Loading classification model...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if device in ["cuda", "mps"] else -1
    )
    print("✅ Classification model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load classification model: {e}")
    raise

# === CLASSIFICATION LABELS ===
researcher_labels = [
    "researcher", "scientist", "computer scientist", "AI researcher", "machine learning researcher",
    "academic", "professor", "engineer", "data scientist", "mathematician", "physicist",
    "chemist", "biologist", "linguist", "university lecturer", "postdoctoral researcher",
    "doctoral student", "technical researcher", "scientific author", "science writer"
]

non_research_labels = [
    "comedian", "actor", "actress", "musician", "singer", "rapper", "sportsman", "athlete",
    "footballer", "basketball player", "tennis player", "politician", "diplomat", "lawyer",
    "businessman", "businesswoman", "entrepreneur", "journalist", "influencer", "youtuber",
    "television presenter", "celebrity", "movie director", "film producer", "screenwriter",
    "artist", "painter", "fashion designer", "model", "novelist", "poet", "author",
    "motivational speaker", "podcaster", "public speaker", "filmmaker"
]

topic_labels = [
    "machine learning", "natural language processing", "computer vision", "data mining",
    "robotics", "theoretical computer science", "cryptography", "bioinformatics",
    "networks", "algorithms", "human-computer interaction", "software engineering",
    "databases", "systems", "security and privacy", "artificial intelligence",
    "deep learning", "neural networks", "reinforcement learning", "optimization",
    "distributed systems", "cloud computing", "quantum computing", "blockchain",
    "Agentic AI", "Embodied AI", "AI safety", "Computer Architecture","Efficient AI","Efficient Systems"
]

researcher_candidate_labels = researcher_labels + non_research_labels

# === CONFIG ===
SAVE_EVERY = 5
MAX_CRAWL_DEPTH = 800000000000
SEED_USER_ID = "kukA0LcAAAAJ"  # Yoshua Bengio
PROGRESS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "scholar_profiles.csv")
GRAPH_GRAPHML = os.path.join(PROJECT_ROOT, "cache", "coauthor_network_progressssss.graphml")
QUEUE_FILE = os.path.join(PROJECT_ROOT, "cache", "queue.txt")
MAX_CRAWL_SECONDS = 3600000

ICLR_PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "iclr_2020_2025.parquet")
NEURIPS_PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "neurips_2020_2024_combined_data.parquet")
FUZZY_RUN_INTERVAL = 5
phrase_to_topics = {}
phrase_to_topics_lock = threading.Lock()  # Thread lock for phrase_to_topics dictionary
fuzzy_cache_lock = threading.Lock()  # Thread lock for fuzzy_cache dictionary

# === BATCH TUNING ===
BATCH_SIZE = 100  # Number of profiles to process per batch
SAVE_INTERVAL = 100  # Number of processed profiles before saving progress

# Load conference data once for fuzzy matching (module scope)
iclr_df = pd.read_parquet(ICLR_PARQUET_PATH) if os.path.exists(ICLR_PARQUET_PATH) else None
neurips_df = pd.read_parquet(NEURIPS_PARQUET_PATH) if os.path.exists(NEURIPS_PARQUET_PATH) else None

EXPECTED_COLUMNS = [
    "user_id", "name", "position", "email", "homepage", "country", "institution", "research_interests",
    "interest_phrases", "citations_all", "h_index_all", "search_depth", "parent_id",
    "Participated_in_ICLR", "wiki_birth_name", "wiki_name", "wiki_birth_date", "wiki_birth_place", 
    "wiki_death_date", "wiki_death_place", "wiki_fields", "wiki_work_institution", "wiki_alma_mater", 
    "wiki_notable_students", "wiki_thesis_title", "wiki_thesis_year", "wiki_thesis_url", "wiki_known_for",
    "wiki_awards", "wiki_deceased", "wiki_wiki_summary", "wiki_is_researcher_ml", "wiki_matched_title", 
    "coauthors", "Fuzzy_Matched", "ICLR_Institution", "Participated_in_NeurIPS", "NeurIPS_Institution"
]

# === MAPPINGS & CONFIGS ===
domain_to_institution = {
    "nus.edu.sg": "National University of Singapore",
    "a-star.edu.sg": "A*STAR (Agency for Science, Technology and Research)",
    "mit.edu": "Massachusetts Institute of Technology",
    "cam.ac.uk": "University of Cambridge",
    "imperial.ac.uk": "Imperial College London",
    "ucl.ac.uk": "University College London",
    "ucla.edu": "University of California, Los Angeles",
    "stanford.edu": "Stanford University",
    "harvard.edu": "Harvard University",
    "berkeley.edu": "University of California, Berkeley",
    "utoronto.ca": "University of Toronto",
    "queensu.ca": "Queen's University",
    "unimelb.edu.au": "University of Melbourne",
    "sydney.edu.au": "University of Sydney",
    "monash.edu": "Monash University",
    "kaust.edu.sa": "King Abdullah University of Science and Technology",
    "tsinghua.edu.cn": "Tsinghua University",
    "pku.edu.cn": "Peking University",
    "ethz.ch": "ETH Zurich",
    "epfl.ch": "École Polytechnique Fédérale de Lausanne",
    "google.com": "Google LLC",
    "microsoft.com": "Microsoft Corporation",
    "ibm.com": "IBM Corporation",
    "amazon.com": "Amazon.com, Inc.",
    "facebook.com": "Meta Platforms, Inc.",
    "openai.com": "OpenAI",
    "flatironinstitute.org": "Flatiron Institute",
    "ed.ac.uk": "University of Edinburgh",
    "bham.ac.uk": "University of Birmingham",
    "ncl.ac.uk": "Newcastle University",
    "manchester.ac.uk": "University of Manchester",
    "lboro.ac.uk": "Loughborough University",
    "deshawresearch.com": "D. E. Shaw Research",
    "cs.ubc.ca": "University of British Columbia",
    "univ-lille.fr": "Universite de Lille",
    "cs.cmu.edu": "Carnegie Mellon University",
    "mcgill.ca": "McGill University",
    "purdue.edu":"Purdue University",
    "Nvidia.com":"Nvidia",
    "ucr.edu": "UC Riverside",
    "dfdxlabs.com": "DFDX Labs",
    "umontreal.ca": "Université de Montréal",
    "ufl.edu": "University of Florida",
    "austin.utexas.edu": "University of Texas at Austin",
    "iro.umontreal.ca": "Université de Montréal",
    "normalesup.org": "École Normale Supérieure",
    "students.edu.sg": "Singapore Education (Students)",
    "buzzj.com": "BuzzJ",
    "fb.com": "Facebook",
    "polimi.it": "Politecnico di Milano",
    "anthropic.com": "Anthropic",
    "cs.wisc.edu": "University of Wisconsin–Madison",
    "cs.mcgill.ca": "McGill University",
    "cohere.com": "Cohere",
    "elementai.com": "Element AI",
    "meta.com": "Meta",
    "nvidia.com": "Nvidia",
    "polymtl.ca": "Polytechnique Montréal",
    "ocado.com": "Ocado",
    "cegepmontpetit.ca": "Cégep Édouard-Montpetit",
    "bottou.org": "Bottou.org",
    "generalagents.com": "General Agents",
    "servicenow.com": "ServiceNow",
    "mila.quebec": "Mila",
    "cs.nyu.edu": "New York University",
    "nyu.edu": "New York University",
    "deepmind.com": "DeepMind",
    "cims.nyu.edu": "New York University",
    "mail.mcgill.ca": "McGill University",
    "dhruvbatra.com": "Dhruv Batra",
    "umich.edu": "University of Michigan",
    "cs.washington.edu": "University of Washington",
    "cise.ufl.edu": "University of Florida",
    "anyscale.com": "Anyscale",
    "apple.com": "Apple",
    "kaist.ac.kr": "KAIST",
    "gatech.edu": "Georgia Institute of Technology",
    "reliant.ai": "Reliant AI",
    "uni.lu": "University of Luxembourg",
    "dauphin.io": "Dauphin.io",
    "jonasschneider.com": "Jonas Schneider",
    "concordia.ca": "Concordia University",
    "hec.ca": "HEC Montréal",
    "cs.ucsc.edu": "University of California, Santa Cruz",
    "cse.psu.edu": "Pennsylvania State University"
}

suffix_country_map = {
    'edu.sg': 'sg',
    'gov.sg': 'sg',
    'ac.uk': 'uk',
    'edu.au': 'au',
    'edu.cn': 'cn',
    'edu.in': 'in',
    'edu.ca': 'ca',
    'edu': 'us',
    'gov': 'us',
    'ac.jp': 'jp',
    'ac.kr': 'kr',
    'ac.at': 'at',
    'ac.be': 'be',
    'ac.nz': 'nz',
    'com': 'unknown',
    'org': 'unknown',
    'net': 'unknown',
    'sg': 'sg',
    'uk': 'uk',
    'us': 'us',
    'fr': 'fr',
    'de': 'de',
    'at': 'at',
    'ca': 'ca',
    'au': 'au',
    'cn': 'cn',
    'jp': 'jp',
    'kr': 'kr',
    "dhruvbatra.com": "us",
    "meta.com": "us",
    "google.com": "us",
    "anyscale.com": "us",
    "apple.com": "us",
    "microsoft.com": "us",
    "reliant.ai": "us",
    "uni.lu": "lu",
    "dauphin.io": "us",         
    "jonasschneider.com": "us", 
    "anthropic.com": "us",
    "openai.com": "us",
    "ibm.com": "us",
    "nvidia.com": "us",
    "dfdxlabs.com": "us",       
    "epfl.ch": "ch",            
    "normalesup.org": "fr",     
    "buzzj.com": "us",          
    "fb.com": "us",             
    "polimi.it": "it",          
    "cohere.com": "ca",         
    "elementai.com": "ca",      
    "amazon.com": "us",
    "ocado.com": "uk",
    "bottou.org": "us",         
    "generalagents.com": "us",
    "servicenow.com": "us",
    "mila.quebec": "ca"         
    # Add more as needed...
}

synonym_map = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
}

multi_word_phrases = [
    "machine learning", "artificial intelligence", "quantum chemistry",
    "computational materials science", "deep learning", "molecular dynamics",
    "homogeneous catalysis", "organometallic chemistry", "polymer chemistry",
    "drug discovery", "genome engineering", "synthetic biology", "protein engineering",
    "metabolic engineering", "quantum computing", "density functional theory"
]

# === CLASSIFICATION FUNCTIONS ===
def classify_researcher_from_summary(summary):
    """Classify if a Wikipedia summary describes a researcher"""
    start = time.time()
    if not summary or not summary.strip():
        return False
    
    try:
        classification = classifier(summary.strip(), researcher_candidate_labels)
        top_label = classification['labels'][0].lower()
        return top_label in [label.lower() for label in researcher_labels]
    except Exception:
        return False

classify_researcher_from_summary = lru_cache(maxsize=256)(classify_researcher_from_summary)

def tag_interest_phrases(phrases):
    """Tag interest phrases with research topics"""
    start = time.time()
    tags = set()
    for phrase in phrases:
        # Thread-safe check and update of phrase_to_topics
        with phrase_to_topics_lock:
            if phrase in phrase_to_topics:
                # Get cached result
                cached_labels = phrase_to_topics[phrase]
                tags.update(cached_labels)
                continue
        
        # Process phrase outside of lock to avoid blocking other threads
        try:
            result = classifier(phrase, topic_labels)
            top_labels = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.3]
        except Exception:
            top_labels = []
        
        # Update dictionary with lock
        with phrase_to_topics_lock:
            # Double-check in case another thread updated it while we were processing
            if phrase not in phrase_to_topics:
                phrase_to_topics[phrase] = top_labels
            cached_labels = phrase_to_topics[phrase]
        
        tags.update(cached_labels)
    return sorted(tags)

# === WIKIPEDIA FUNCTIONS ===
def clean_wiki_markup(raw_text):
    """Clean Wikipedia markup and return plain text"""
    start = time.time()
    wikicode = mwparserfromhell.parse(raw_text)
    
    # Handle special templates
    for template in wikicode.filter_templates():
        if template.name.lower().strip() in ['ubl', 'plainlist', 'flatlist', 'hlist']:
            items = [mwparserfromhell.parse(str(param.value)).strip_code().strip() for param in template.params]
            items = [re.sub(r'\\[\\[]|\\[\\]]', '', item) for item in items if item]
            wikicode.replace(template, "; ".join(items))
    
    # Remove links and templates
    for link in wikicode.filter_wikilinks():
        link_text = link.text if link.text else link.title
        wikicode.replace(link, str(link_text))
    
    for template in wikicode.filter_templates():
        wikicode.remove(template)
    
    # Clean text
    cleaned_text = wikicode.strip_code()
    cleaned_text = re.sub(r';+\s*', '; ', cleaned_text.replace('\n', '; ')).strip()
    cleaned_text = re.sub(r'[{}\|]', '', cleaned_text).strip('; ').strip()
    return cleaned_text

def fuzzy_wikipedia_search(name, threshold=0.90, max_results=5):
    """Search Wikipedia with fuzzy matching"""
    start = time.time()
    S = requests.Session()
    S.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; WikiInfoBot/1.0)'})
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {"action": "query", "list": "search", "srsearch": name, "srlimit": max_results, "format": "json"}
    
    for _ in range(3):  # Retry logic
        response = S.get(url=URL, params=PARAMS)
        if response.status_code == 200:
            try:
                data = response.json()
                break
            except Exception:
                time.sleep(1)
    else:
        return None
    
    search_results = data.get("query", {}).get("search", [])
    best_match, best_score = None, 0
    
    for result in search_results:
        title = result['title']
        score = difflib.SequenceMatcher(None, name.lower(), title.lower()).ratio()
        if score > best_score:
            best_score, best_match = score, title
    return best_match if best_score >= threshold else None


# === ASYNC WIKIPEDIA FUNCTIONS ===
async def async_get_wikipedia_summary(session, page_title):
    """Async get Wikipedia page summary"""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; WikiInfoBot/1.0)'}
    for _ in range(3):
        try:
            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("extract", "")
                else:
                    await asyncio.sleep(1)
        except asyncio.TimeoutError:
            await asyncio.sleep(2)
        except Exception:
            await asyncio.sleep(1)
    return ""


async def async_get_selected_infobox_fields(session, page_title, fields_to_extract):
    """Async extract specific fields from Wikipedia infobox"""
    URL = "https://en.wikipedia.org/w/api.php"
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; WikiInfoBot/1.0)'}
    PARAMS = {"action": "query", "format": "json", "titles": page_title,
              "prop": "revisions", "rvprop": "content", "rvslots": "main"}
    for _ in range(3):
        try:
            async with session.get(URL, params=PARAMS, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    break
                else:
                    await asyncio.sleep(1)
        except asyncio.TimeoutError:
            await asyncio.sleep(2)
        except Exception:
            await asyncio.sleep(1)
    else:
        return None, None
    page = next(iter(data['query']['pages'].values()))
    if "missing" in page or 'revisions' not in page:
        return None, None
    matched_title = page.get('title', page_title)
    wikitext = page['revisions'][0]['slots']['main']['*']
    wikicode = mwparserfromhell.parse(wikitext)
    infobox = next((t for t in wikicode.filter_templates() if t.name.lower().strip().startswith("infobox")), None)
    extracted = {}
    if infobox:
        for key in fields_to_extract:
            try:
                val = infobox.get(key).value.strip()
                extracted[key] = clean_wiki_markup(str(val))
            except Exception:
                extracted[key] = ""
    else:
        extracted = {k: "" for k in fields_to_extract}
    return extracted, matched_title


async def async_get_author_wikipedia_info(session, author_name):
    """Async get comprehensive Wikipedia info for an author"""
    fields = ["birth_name", "name", "birth_date", "birth_place", "death_date", "death_place",
              "fields", "work_institution", "alma_mater", "notable_students", "thesis_title",
              "thesis_year", "thesis_url", "known_for", "awards"]
    # Persistent cache lookup
    if author_name in wiki_cache:
        return wiki_cache[author_name]
    matched_title = fuzzy_wikipedia_search(author_name)
    if matched_title is None:
        info = {k: "" for k in fields}
        info.update({"deceased": "False", "wiki_summary": "", "is_researcher_ml": "False", "matched_title": ""})
        wiki_cache[author_name] = info
        joblib.dump(wiki_cache, WIKI_CACHE_PATH)
        return info
    info, matched_title = await async_get_selected_infobox_fields(session, matched_title, fields)
    if info is None:
        info = {k: "" for k in fields}
        info["deceased"] = "False"
    else:
        info["deceased"] = str(bool(info.get("death_date")))
    summary = await async_get_wikipedia_summary(session, matched_title)
    info.update({
        "wiki_summary": summary,
        "is_researcher_ml": str(classify_researcher_from_summary(summary)),
        "matched_title": matched_title or ""
    })
    wiki_cache[author_name] = info
    joblib.dump(wiki_cache, WIKI_CACHE_PATH)
    return info

# === UTILITY FUNCTIONS ===
def normalize_interest_phrases(raw_text):
    """Normalize and process interest phrases"""
    interests = [s.strip().lower() for s in raw_text.split(",") if s.strip()]
    processed = []
    for interest in interests:
        interest = synonym_map.get(interest, interest)
        for phrase in multi_word_phrases:
            if interest == phrase:
                interest = phrase.replace(" ", "_")
        processed.append(interest)
    return processed

def get_driver():
    """Initialize headless Chrome driver with better error handling"""
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--single-process")
        options.add_argument("--no-zygote")
        print("🔧 Initializing Chrome driver...")
        driver = webdriver.Chrome(options=options)
        print("✅ Chrome driver initialized successfully")
        return driver
    except Exception as e:
        print(f"❌ Failed to initialize Chrome driver: {e}")
        raise

def infer_country_from_email_field(email_field):
    """Extract country from email field"""
    match = re.search(r"Verified email at ([^\s]+?)(?:\s*-\s*Homepage)?$", email_field)
    if not match:
        return "unknown"
    
    domain = match.group(1).lower().strip()
    ext = tldextract.extract(domain)
    suffix = ext.suffix.lower()
    
    if suffix in suffix_country_map:
        return suffix_country_map[suffix]
    else:
        last_part = suffix.split('.')[-1]
        return suffix_country_map.get(last_part, "unknown")

def get_institution_from_email(email_field):
    """Extract institution from email field"""
    match = re.search(r"Verified email at ([^\s]+?)(?:\s*-\s*Homepage)?$", email_field)
    if not match:
        return "Unknown"
    
    domain = match.group(1).lower().strip()
    for known_domain in domain_to_institution:
        if domain.endswith(known_domain):
            return domain_to_institution[known_domain]
    return "Unknown"

def extract_profile(driver, user_id, depth, parent_id=None):
    start = time.time()
    """Extract complete profile from Google Scholar, using requests+BeautifulSoup if possible, else fallback to Selenium."""
    url = f"https://scholar.google.com/citations?hl=en&user={user_id}"
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; ScholarScraper/1.0)'}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200 and 'gsc_prf_in' in resp.text:
            soup = BeautifulSoup(resp.text, "html.parser")
        else:
            raise Exception("Profile not found or blocked")
    except Exception:
        # Fallback to Selenium if requests fails or is blocked
        driver.get(url)
        try:
            WebDriverWait(driver, 1).until(
                EC.presence_of_element_located((By.ID, "gsc_prf_in"))
            )
        except Exception as e:
            print(f"⚠️ Wait timeout for {url} — {e}")
            time.sleep(0.5)
        soup = BeautifulSoup(driver.page_source, "html.parser")

    # Extract basic info
    name_tag = soup.select_one("#gsc_prf_in")
    position_tag = soup.select_one(".gsc_prf_il")
    email_tag = soup.select_one("#gsc_prf_ivh")
    interest_tags = soup.select("#gsc_prf_int a")
    metrics_rows = soup.select("#gsc_rsb_st tbody tr")

    name = name_tag.get_text(strip=True) if name_tag else "Unknown"
    position = position_tag.get_text(strip=True) if position_tag else "Unknown"
    email = email_tag.get_text(strip=True) if email_tag else "Unknown"
    interests_raw = ", ".join(tag.text for tag in interest_tags) if interest_tags else ""
    interest_phrases = normalize_interest_phrases(interests_raw)
    country = infer_country_from_email_field(email)
    institution = get_institution_from_email(email)

    # Extract homepage
    homepage_url = ""
    if email_tag:
        homepage_link = email_tag.find("a", string="Homepage") or email_tag.find("a")
        from bs4.element import Tag
        if homepage_link and isinstance(homepage_link, Tag) and homepage_link.get_text(strip=True).lower() == "homepage":
            homepage_url = homepage_link.get("href", "")

    # Extract metrics
    citations_all = "0"
    h_index_all = "0"
    if metrics_rows and len(metrics_rows) >= 2:
        try:
            citations_all = metrics_rows[0].select("td")[1].text
            h_index_all = metrics_rows[1].select("td")[1].text
        except Exception:
            pass

    # Get coauthors
    coauthors = []
    for a_tag in soup.select(".gsc_rsb_aa .gsc_rsb_a_desc a"):
        href = a_tag.get("href", "")
        if isinstance(href, list):
            href = href[0] if href else ""
        if isinstance(href, str) and "user=" in href:
            co_id = href.split("user=")[1].split("&")[0]
            if co_id != user_id:
                coauthors.append(co_id)

    # Add parent to coauthors if exists
    if parent_id and parent_id != user_id and parent_id not in coauthors:
        coauthors.append(parent_id)

    return {
        "user_id": user_id, "name": name, "position": position, "email": email,
        "homepage": homepage_url, "country": country, "institution": institution,
        "research_interests": interests_raw, "interest_phrases": interest_phrases,
        "citations_all": citations_all, "h_index_all": h_index_all,
        "search_depth": depth, "parent_id": parent_id, "coauthors": coauthors,
    }

# === FILE MANAGEMENT ===
def ensure_progress_csv(path):
    """Ensure progress CSV exists with proper columns"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(path, index=False)
    else:
        try:
            df = pd.read_csv(path, low_memory=False)
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            if missing_cols:
                pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(path, index=False)
        except Exception:
            pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(path, index=False)

def save_progress(profiles, _graph=None):
    """Save profiles to CSV"""
    df = pd.DataFrame(profiles)
    df.to_csv(PROGRESS_CSV, index=False)

# === QUEUE MANAGEMENT ===
def load_queue():
    """Load queue from file"""
    if not os.path.exists(QUEUE_FILE):
        return deque()
    
    with open(QUEUE_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    queue = deque()
    for line in lines:
        try:
            data = json.loads(line)
            queue.append(tuple(data))
        except json.JSONDecodeError:
            queue.append((line, 0, None))  # Legacy format
    return queue

def save_queue(queue):
    """Save queue to file"""
    with open(QUEUE_FILE, "w") as f:
        for item in queue:
            f.write(json.dumps(item) + "\n")

def enqueue_user(queue, user_id, depth, parent_id=None, visited_ids=None, prepend=False):
    """Add user to queue if not already present"""
    if visited_ids is None:
        visited_ids = set()
    if user_id in visited_ids or any(user_id == item[0] for item in queue):
        return
    
    new_item = (user_id, depth, parent_id)
    if prepend:
        queue.appendleft(new_item)
    else:
        queue.append(new_item)

# === FUZZY MATCHING ===
def fuzzy_match_conference_participation(profile, conf_name, df, name_col='Author', inst_col='Institution', threshold=85):
    """Match profile to conference participation"""
    authors = df[name_col].dropna().unique()
    authors_lower = pd.Series(authors).str.lower().values

    profile_name = profile.get("name", "").lower()
    if not profile_name:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Institution"] = ""
        return

    # Persistent cache for fuzzy match with thread safety
    cache_key = f"{conf_name}:{profile_name}"
    with fuzzy_cache_lock:
        if cache_key in fuzzy_cache:
            match, score, matched_institution = fuzzy_cache[cache_key]
            if float(score) >= threshold:
                profile[f"Participated_in_{conf_name}"] = True
                profile[f"{conf_name}_Institution"] = matched_institution
            else:
                profile[f"Participated_in_{conf_name}"] = False
                profile[f"{conf_name}_Institution"] = ""
            return

    result = process.extractOne(profile_name, authors_lower, scorer=fuzz.token_sort_ratio)
    if result is None:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Institution"] = ""
        return
    
    match, score = result[0], result[1]
    if score >= threshold:
        # Vectorized lookup for matched institution
        mask = (authors_lower == match)
        matched_author = authors[mask][0] if mask.any() else ""
        if inst_col in df.columns:
            inst_values = df.loc[df[name_col] == matched_author, inst_col].dropna()
            matched_institution = inst_values.iloc[0] if not inst_values.empty else ""
        else:
            matched_institution = ""
        profile[f"Participated_in_{conf_name}"] = True
        profile[f"{conf_name}_Institution"] = matched_institution
    else:
        matched_institution = ""
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Institution"] = ""
    
    # Update cache with thread safety
    with fuzzy_cache_lock:
        fuzzy_cache[cache_key] = (str(match), str(score), str(matched_institution))
        joblib.dump(fuzzy_cache, FUZZY_CACHE_PATH)

def run_fuzzy_matching_single(profile):
    """Run fuzzy matching for a single profile"""
    if profile.get("Fuzzy_Matched"):
        return

    # Use preloaded DataFrames for fuzzy matching
    if iclr_df is not None:
        fuzzy_match_conference_participation(profile, "ICLR", iclr_df)
    if neurips_df is not None:
        fuzzy_match_conference_participation(profile, "NeurIPS", neurips_df)
    profile["Fuzzy_Matched"] = True

# === PROGRESS LOADING ===
def load_progress():
    """Load all progress from files"""
    all_profiles = []
    visited_depths = {}
    graph = nx.Graph()
    all_interest_phrases = defaultdict(list)

    # Load profiles
    if os.path.exists(PROGRESS_CSV):
        try:
            profiles_df = pd.read_csv(PROGRESS_CSV, low_memory=False)
            all_profiles = profiles_df.to_dict(orient="records")
            for p in all_profiles:
                visited_depths[p["user_id"]] = p.get("search_depth", 0)
        except Exception as e:
            print(f"⚠️ Error loading profiles: {e}")

    # Load graph
    if os.path.exists(GRAPH_GRAPHML):
        try:
            graph = nx.read_graphml(GRAPH_GRAPHML)
        except Exception:
            graph = nx.Graph()

    # Rebuild interest phrases
    for p in all_profiles:
        for phrase in p.get("interest_phrases", []):
            if phrase:
                all_interest_phrases[phrase].append(p["user_id"])

    return all_profiles, visited_depths, graph, all_interest_phrases

def get_coauthors_from_profile(driver, user_id):
    """Get coauthors for a specific profile"""
    url = f"https://scholar.google.com/citations?hl=en&user={user_id}"
    driver.get(url)
    time.sleep(random.uniform(1.5, 3.0))
    soup = BeautifulSoup(driver.page_source, "html.parser")

    coauthors = []
    for a_tag in soup.select(".gsc_rsb_aa .gsc_rsb_a_desc a"):
        href = a_tag.get("href", "")
        if isinstance(href, list):
            href = href[0] if href else ""
        if isinstance(href, str) and "user=" in href:
            co_id = href.split("user=")[1].split("&")[0]
            coauthors.append(co_id)
    return coauthors


def enrich_profile_base(profile, wiki_info):
    try:
        # Tag interests
        if "interest_phrases" in profile and profile["interest_phrases"]:
            profile["topic_tags"] = tag_interest_phrases(profile["interest_phrases"])
        else:
            profile["topic_tags"] = []
        # Fuzzy matching
        run_fuzzy_matching_single(profile)
        # Wikipedia info (already fetched)
        profile.update({
            f"wiki_{k}": v for k, v in wiki_info.items()
        })
        return profile
    except Exception as e:
        print(f"❌ Enrichment failed for {profile.get('user_id')}: {e}")
        return profile

# Batch async Wikipedia enrichment
async def batch_enrich_profiles(profiles):
    async with aiohttp.ClientSession() as session:
        tasks = [async_get_author_wikipedia_info(session, p.get("name", "")) for p in profiles]
        wiki_infos = await asyncio.gather(*tasks)
    # Now do the rest of enrichment (topic tags, fuzzy matching) in parallel
    from concurrent.futures import ThreadPoolExecutor
    enriched = []
    with ThreadPoolExecutor() as executor:
        enriched = list(executor.map(lambda args: enrich_profile_base(*args), zip(profiles, wiki_infos)))
    return enriched

# === MAIN CRAWLING FUNCTION ===
def crawl_bfs_resume(driver, queue, all_profiles, visited_depths, force=False):
    """Main BFS crawling function with batched multiprocessing enrichment"""
    start_time = time.time()
    total_scraped = 0
    INSERT_FILE = "queue_insert.jsonl"
    INSERT_CHECK_INTERVAL = 30
    last_insert_check = time.time()

    visited_ids = set(p.get("user_id") for p in all_profiles if isinstance(p, dict))
    if force:
        queued_user_ids = set(item[0] for item in queue)
        for uid in queued_user_ids:
            visited_depths.pop(uid, None)

    profile_batch = []
    batch_since_save = 0

    # Convert visited_depths to set for O(1) lookup
    visited_set = set(visited_depths.keys())

    while queue:
        if time.time() - last_insert_check > INSERT_CHECK_INTERVAL:
            if os.path.exists(INSERT_FILE):
                try:
                    with open(INSERT_FILE, "r") as f:
                        new_lines = [line.strip() for line in f if line.strip()]
                    os.remove(INSERT_FILE)
                    new_items = []
                    for line in new_lines:
                        try:
                            user_id, depth, parent = json.loads(line)
                            if user_id not in visited_set and not any(user_id == q[0] for q in queue):
                                new_items.append((user_id, depth, parent))
                        except Exception:
                            continue
                    if new_items:
                        for item in reversed(new_items):
                            queue.appendleft(item)
                except Exception:
                    pass
            last_insert_check = time.time()

        # Parallelize extraction of multiple profiles at once
        batch_items = []
        while queue and len(batch_items) < BATCH_SIZE:
            user_id, depth, parent_id = queue.popleft()
            depth = int(depth)
            if not force and user_id in visited_set:
                continue
            batch_items.append((user_id, depth, parent_id))
        if not batch_items:
            continue
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            profiles = list(executor.map(lambda args: extract_profile(driver, *args), batch_items))
        for profile in profiles:
            visited_depths[profile["user_id"]] = profile["search_depth"]
            visited_set.add(profile["user_id"])
            profile_batch.append(profile)
            total_scraped += 1
            # Enqueue coauthors
            for co_id in profile.get("coauthors", [])[:20]:
                if co_id not in visited_set and co_id != profile["user_id"]:
                    enqueue_user(queue, co_id, profile["search_depth"] + 1, profile["user_id"])
        # Process batch if size reached
        if len(profile_batch) >= BATCH_SIZE:
            batch_start = time.time()
            enriched_profiles = asyncio.run(batch_enrich_profiles(profile_batch))
            batch_time = time.time() - batch_start
            all_profiles.extend(enriched_profiles)
            profile_batch = []
            batch_since_save += len(enriched_profiles)
            print(f"✅ Processed {total_scraped} profiles so far | Batch size: {BATCH_SIZE} | Time: {batch_time:.2f}s")
            progress_bar.update(len(enriched_profiles))
            # Only save after each batch, not after every profile
            if batch_since_save >= SAVE_INTERVAL:
                save_queue(queue)
                save_progress(all_profiles)
                batch_since_save = 0

    # Flush leftover profiles after queue emptied
    if profile_batch:
        enriched_profiles = asyncio.run(batch_enrich_profiles(profile_batch))
        all_profiles.extend(enriched_profiles)
        profile_batch = []
        save_progress(all_profiles)

    save_queue(queue)
    save_progress(all_profiles)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    try:
        print("🚀 Starting Scholar Scraper...")
        
        # Initialize globals
        phrase_to_topics = {}
        print("🔧 Initializing Chrome driver...")
        driver = get_driver()
        progress_bar = tqdm(total=0, desc="Crawling profiles", dynamic_ncols=True)

        print("📊 Loading progress...")
        # Load progress
        all_profiles, visited_depths, graph, all_interest_phrases = load_progress()
        ensure_progress_csv(PROGRESS_CSV)
        print(f"📈 Loaded {len(all_profiles)} existing profiles")

        # === Initialize queue with multi-layer fallback ===
        print("📋 Initializing queue...")
        if os.path.exists(QUEUE_FILE):
            queue = load_queue()
            print(f"📥 Loaded {len(queue)} items from queue")
        else:
            print("🔄 Creating new queue...")
            queue = deque()
            if all_profiles:
                last_users = [p["user_id"] for p in all_profiles[-10:] if p.get("user_id")]
                if last_users:
                    max_depth = max(visited_depths.get(uid, 0) for uid in last_users)
        
                    coauthors = set()
                    for user_id in last_users:
                        try:
                            new_coauthors = get_coauthors_from_profile(driver, user_id)
                            for co in new_coauthors:
                                if co not in visited_depths:
                                    coauthors.add(co)
                        except Exception as e:
                            print(f"⚠️ Error getting coauthors for {user_id}: {e}")
                            continue
        
                    if coauthors:
                        for co in coauthors:
                            queue.append((co, max_depth + 1, None))
                    else:
                        queue.append((SEED_USER_ID, 0, None))
                else:
                    queue.append((SEED_USER_ID, 0, None))
            else:
                queue.append((SEED_USER_ID, 0, None))
        
        print(f"🎯 Starting crawl with {len(queue)} items in queue...")
        crawl_bfs_resume(driver, queue, all_profiles, visited_depths, force=True)
        print("✅ Crawling complete!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🧹 Cleaning up...")
        try:
            progress_bar.close()
        except:
            pass
        try:
            driver.quit()
            print("✅ Chrome driver closed")
        except:
            pass