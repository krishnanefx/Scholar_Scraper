import os
import time
import random
import json
import re
import pandas as pd
import tldextract
import requests
import mwparserfromhell
import difflib
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from fuzzywuzzy import process, fuzz
from collections import Counter, defaultdict, deque
from transformers import pipeline
import torch

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === CONFIGURATION ===
SAVE_EVERY = 5
MAX_CRAWL_DEPTH = 800000000000
SEED_USER_ID = "0b_Q5gcAAAAJ"
PROGRESS_CSV = "scholar_profiles_progressssss.csv"
GRAPH_GRAPHML = "coauthor_network_progressssss.graphml"
QUEUE_FILE = "queue.txt"
NUM_PHRASE_CLUSTERS = 20
MAX_CRAWL_SECONDS = 3600000  # 1 hour
FUZZY_CACHE_PATH = "fuzzy_match_cache.json"
ICLR_PARQUET_PATH = 'iclr_2020_2025_combined_data.parquet'
NEURIPS_PARQUET_PATH = 'neurips_2020_2024_combined_data.parquet'
FUZZY_RUN_INTERVAL = 5
INSERT_FILE = "queue_insert.jsonl"
INSERT_CHECK_INTERVAL = 30

# Fixed: Added missing ICLR_Matched_Name column
EXPECTED_COLUMNS = [
    "user_id", "name", "position", "email", "country", "institution", "research_interests",
    "interest_phrases", "citations_all", "h_index_all", "topic_clusters", "search_depth",
    "Participated_in_ICLR", "ICLR_Institution", "Participated_in_NeurIPS", 
    "NeurIPS_Institution", "Fuzzy_Matched", "wiki_birth_name", "wiki_name", "wiki_birth_date", 
    "wiki_birth_place", "wiki_death_date", "wiki_death_place", "wiki_fields", 
    "wiki_work_institution", "wiki_alma_mater", "wiki_notable_students", "wiki_thesis_title",
    "wiki_thesis_year", "wiki_thesis_url", "wiki_known_for", "wiki_awards", "wiki_deceased", 
    "wiki_wiki_summary", "wiki_is_researcher_ml", "wiki_matched_title", "coauthors"
]

# === DOMAIN MAPPINGS ===
DOMAIN_TO_INSTITUTION = {
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
}

SUFFIX_COUNTRY_MAP = {
    'edu.sg': 'sg', 'gov.sg': 'sg', 'ac.uk': 'uk', 'edu.au': 'au', 'edu.cn': 'cn',
    'edu.in': 'in', 'edu.ca': 'ca', 'edu': 'us', 'gov': 'us', 'ac.jp': 'jp',
    'ac.kr': 'kr', 'ac.at': 'at', 'ac.be': 'be', 'ac.nz': 'nz', 'com': 'unknown',
    'org': 'unknown', 'net': 'unknown', 'sg': 'sg', 'uk': 'uk', 'us': 'us',
    'fr': 'fr', 'de': 'de', 'at': 'at', 'ca': 'ca', 'au': 'au', 'cn': 'cn',
    'jp': 'jp', 'kr': 'kr'
}

SYNONYM_MAP = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
}

MULTI_WORD_PHRASES = [
    "machine learning", "artificial intelligence", "quantum chemistry",
    "computational materials science", "deep learning", "molecular dynamics",
    "homogeneous catalysis", "organometallic chemistry", "polymer chemistry",
    "drug discovery", "genome engineering", "synthetic biology", "protein engineering",
    "metabolic engineering", "quantum computing", "density functional theory"
]

# === ML MODEL SETUP ===
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

device = get_device()
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if device in ["cuda", "mps"] else -1
)

RESEARCHER_LABELS = [
    "researcher", "scientist", "computer scientist", "AI researcher", "machine learning researcher",
    "academic", "professor", "engineer", "data scientist", "mathematician", "physicist",
    "chemist", "biologist", "linguist", "university lecturer", "postdoctoral researcher",
    "doctoral student", "technical researcher", "scientific author", "science writer"
]

NON_RESEARCH_LABELS = [
    "comedian", "actor", "actress", "musician", "singer", "rapper", "sportsman", "athlete",
    "footballer", "basketball player", "tennis player", "politician", "diplomat", "lawyer",
    "businessman", "businesswoman", "entrepreneur", "journalist", "influencer", "youtuber",
    "television presenter", "celebrity", "movie director", "film producer", "screenwriter",
    "artist", "painter", "fashion designer", "model", "novelist", "poet", "author",
    "motivational speaker", "podcaster", "public speaker", "filmmaker"
]

CANDIDATE_LABELS = RESEARCHER_LABELS + NON_RESEARCH_LABELS

# === UTILITY FUNCTIONS ===
def ensure_progress_csv(path):
    """Initialize CSV with proper headers if missing or corrupted"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print("🆕 Initializing empty progress CSV with headers...")
        pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(path, index=False)
    else:
        try:
            df = pd.read_csv(path)
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            if missing_cols:
                print(f"⚠️ CSV missing columns: {missing_cols} — reinitializing.")
                pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(path, index=False)
        except Exception as e:
            print(f"❌ Error reading progress CSV: {e} — reinitializing.")
            pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(path, index=False)

def get_driver():
    """Create and configure Chrome WebDriver"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)

def normalize_interest_phrases(raw_text):
    """Normalize research interest phrases"""
    interests = [s.strip().lower() for s in raw_text.split(",") if s.strip()]
    processed = []
    for interest in interests:
        interest = SYNONYM_MAP.get(interest, interest)
        for phrase in MULTI_WORD_PHRASES:
            if interest == phrase:
                interest = phrase.replace(" ", "_")
        processed.append(interest)
    return processed

def infer_country_from_email_field(email_field):
    """Extract country from email domain"""
    match = re.search(r"Verified email at ([^\s]+?)(?:\s*-\s*Homepage)?$", email_field)
    if not match:
        return "unknown"
    domain = match.group(1).lower().strip()
    ext = tldextract.extract(domain)
    suffix = ext.suffix.lower()
    if suffix in SUFFIX_COUNTRY_MAP:
        return SUFFIX_COUNTRY_MAP[suffix]
    else:
        last_part = suffix.split('.')[-1]
        return SUFFIX_COUNTRY_MAP.get(last_part, "unknown")

def get_institution_from_email(email_field):
    """Extract institution from email domain"""
    match = re.search(r"Verified email at ([^\s]+?)(?:\s*-\s*Homepage)?$", email_field)
    if not match:
        return "Unknown"
    domain = match.group(1).lower().strip()
    for known_domain in DOMAIN_TO_INSTITUTION:
        if domain.endswith(known_domain):
            return DOMAIN_TO_INSTITUTION[known_domain]
    return "Unknown"

def save_progress(profiles, graph=None):
    """Save progress to CSV"""
    df = pd.DataFrame(profiles)
    df.to_csv(PROGRESS_CSV, index=False)
    print(f"💾 Progress saved: {len(profiles)} profiles")

# === WIKIPEDIA INTEGRATION ===
def clean_wiki_markup(raw_text):
    """Clean Wikipedia markup text"""
    wikicode = mwparserfromhell.parse(raw_text)
    for template in wikicode.filter_templates():
        if template.name.lower().strip() in ['ubl', 'plainlist', 'flatlist', 'hlist']:
            items = [mwparserfromhell.parse(str(param.value)).strip_code().strip() for param in template.params]
            items = [re.sub(r'\\[\\[]|\\[\\]]', '', item) for item in items if item]
            wikicode.replace(template, "; ".join(items))
    for link in wikicode.filter_wikilinks():
        link_text = link.text if link.text else link.title
        wikicode.replace(link, str(link_text))
    for template in wikicode.filter_templates():
        wikicode.remove(template)
    cleaned_text = wikicode.strip_code()
    cleaned_text = re.sub(r';+\s*', '; ', cleaned_text.replace('\n', '; ')).strip()
    cleaned_text = re.sub(r'[{}\|]', '', cleaned_text).strip('; ').strip()
    return cleaned_text

def fuzzy_wikipedia_search(name, threshold=0.90, max_results=5):
    """Search Wikipedia for best matching page"""
    S = requests.Session()
    S.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; WikiInfoBot/1.0)'})
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {"action": "query", "list": "search", "srsearch": name, "srlimit": max_results, "format": "json"}
    
    for _ in range(3):  # retries
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

def get_wikipedia_summary(page_title):
    """Get Wikipedia page summary"""
    S = requests.Session()
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"
    S.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; WikiInfoBot/1.0)'})
    response = S.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("extract", "")
    return ""

def get_selected_infobox_fields(page_title, fields_to_extract):
    """Extract specific fields from Wikipedia infobox"""
    S = requests.Session()
    S.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; WikiInfoBot/1.0)'})
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {"action": "query", "format": "json", "titles": page_title,
              "prop": "revisions", "rvprop": "content", "rvslots": "main"}
    
    for _ in range(3):  # retries
        response = S.get(url=URL, params=PARAMS)
        if response.status_code == 200:
            try:
                data = response.json()
                break
            except Exception:
                time.sleep(1)
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

def classify_summary(summary):
    """Classify if person is a researcher based on summary"""
    summary = summary.strip()
    if not summary:
        return False
    classification = classifier(summary, CANDIDATE_LABELS)
    top_label = classification['labels'][0].lower()
    return top_label in [label.lower() for label in RESEARCHER_LABELS]

def get_author_wikipedia_info(author_name):
    """Get comprehensive Wikipedia information for an author"""
    fields = ["birth_name", "name", "birth_date", "birth_place", "death_date", "death_place",
              "fields", "work_institution", "alma_mater", "notable_students", "thesis_title",
              "thesis_year", "thesis_url", "known_for", "awards"]

    matched_title = fuzzy_wikipedia_search(author_name)
    if matched_title is None:
        info = {k: "" for k in fields}
        info["deceased"] = False
        info["wiki_summary"] = ""
        info["is_researcher_ml"] = False
        info["matched_title"] = None
        return info

    info, matched_title = get_selected_infobox_fields(matched_title, fields)
    if info is None:
        info = {k: "" for k in fields}
        info["deceased"] = False
    else:
        info["deceased"] = bool(info.get("death_date"))
    
    summary = get_wikipedia_summary(matched_title)
    info["wiki_summary"] = summary
    info["is_researcher_ml"] = classify_summary(summary)
    info["matched_title"] = matched_title
    return info

# === PROFILE EXTRACTION ===
def extract_profile(driver, user_id, depth):
    """Extract complete profile from Google Scholar"""
    url = f"https://scholar.google.com/citations?hl=en&user={user_id}"
    driver.get(url)
    time.sleep(random.uniform(1.5, 3.0))
    soup = BeautifulSoup(driver.page_source, "html.parser")

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
        if "user=" in href:
            co_id = href.split("user=")[1].split("&")[0]
            coauthors.append(co_id)

    return {
        "user_id": user_id,
        "name": name,
        "position": position,
        "email": email,
        "country": country,
        "institution": institution,
        "research_interests": interests_raw,
        "interest_phrases": interest_phrases,
        "citations_all": citations_all,
        "h_index_all": h_index_all,
        "topic_clusters": [],
        "search_depth": depth,
        "coauthors": coauthors,
    }

def get_coauthors_from_profile(driver, user_id):
    """Extract coauthors from a profile"""
    url = f"https://scholar.google.com/citations?hl=en&user={user_id}"
    driver.get(url)
    time.sleep(random.uniform(1.5, 3.0))
    soup = BeautifulSoup(driver.page_source, "html.parser")

    coauthors = []
    for a_tag in soup.select(".gsc_rsb_aa .gsc_rsb_a_desc a"):
        href = a_tag.get("href", "")
        if "user=" in href:
            co_id = href.split("user=")[1].split("&")[0]
            coauthors.append(co_id)
    print(f"✅ Found {len(coauthors)} co-authors for {user_id}")
    return coauthors

# === FUZZY MATCHING ===
def fuzzy_match_conference_participation(profile, conf_name, df, name_col='Author', inst_col='Institution', threshold=85):
    """Match profile against conference participation data"""
    authors = df[name_col].dropna().unique()
    authors_lower = [a.lower() for a in authors]

    profile_name = profile.get("name", "").lower()
    if not profile_name:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Matched_Name"] = ""
        profile[f"{conf_name}_Institution"] = ""
        return

    match, score = process.extractOne(profile_name, authors_lower, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        match_idx = authors_lower.index(match)
        matched_author = authors[match_idx]
        matched_institution = df[df[name_col] == matched_author][inst_col].dropna().iloc[0] if inst_col in df.columns else ""
        profile[f"Participated_in_{conf_name}"] = True
        profile[f"{conf_name}_Matched_Name"] = matched_author
        profile[f"{conf_name}_Institution"] = matched_institution
        print(f"✅ [{conf_name}] Match: {profile['name']} → {matched_author} @ {matched_institution}")
    else:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Matched_Name"] = ""
        profile[f"{conf_name}_Institution"] = ""
        print(f"❌ [{conf_name}] No match for {profile['name']}")

def run_fuzzy_matching_single(profile):
    """Run fuzzy matching for a single profile"""
    if profile.get("Fuzzy_Matched"):
        return

    if os.path.exists(ICLR_PARQUET_PATH):
        iclr_df = pd.read_parquet(ICLR_PARQUET_PATH)
        fuzzy_match_conference_participation(profile, "ICLR", iclr_df)
    else:
        print("⚠️ ICLR Parquet not found")
        profile["Participated_in_ICLR"] = False
        profile["ICLR_Matched_Name"] = ""
        profile["ICLR_Institution"] = ""

    if os.path.exists(NEURIPS_PARQUET_PATH):
        neurips_df = pd.read_parquet(NEURIPS_PARQUET_PATH)
        fuzzy_match_conference_participation(profile, "NeurIPS", neurips_df)
    else:
        print("⚠️ NeurIPS Parquet not found")
        profile["Participated_in_NeurIPS"] = False
        profile["NeurIPS_Institution"] = ""

    profile["Fuzzy_Matched"] = True

# === QUEUE MANAGEMENT ===
def load_queue():
    """Load crawling queue from file"""
    if not os.path.exists(QUEUE_FILE):
        return deque()
    with open(QUEUE_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    queue = deque()
    for line in lines:
        try:
            data = json.loads(line)  # expects [user_id, depth, parent_id]
            queue.append(tuple(data))
        except json.JSONDecodeError:
            # fallback for legacy format (just user_id string)
            queue.append((line, 0, None))
    return queue

def save_queue(queue):
    """Save crawling queue to file"""
    with open(QUEUE_FILE, "w") as f:
        for item in queue:
            f.write(json.dumps(item) + "\n")

def enqueue_user(queue, user_id, depth, parent_id=None, visited_ids=None, prepend=False):
    """Add user to crawling queue"""
    if visited_ids is None:
        visited_ids = set()
    if user_id in visited_ids:
        return
    if any(user_id == item[0] for item in queue):
        return
    new_item = (user_id, depth, parent_id)
    if prepend:
        queue.appendleft(new_item)
    else:
        queue.append(new_item)

# === DATA LOADING ===
def load_progress():
    """Load all progress from files"""
    all_profiles = []
    visited_depths = {}
    graph = nx.Graph()
    all_interest_phrases = defaultdict(list)

    # Load profiles CSV
    if os.path.exists(PROGRESS_CSV):
        try:
            profiles_df = pd.read_csv(PROGRESS_CSV)
            all_profiles = profiles_df.to_dict(orient="records")
            for p in all_profiles:
                visited_depths[p["user_id"]] = p.get("search_depth", 0)
        except Exception as e:
            print(f"⚠️ Error loading profiles CSV: {e}")

    # Load GraphML safely
    if os.path.exists(GRAPH_GRAPHML):
        try:
            graph = nx.read_graphml(GRAPH_GRAPHML)
        except Exception as e:
            print(f"⚠️ Error loading graphml file: {e}")
            print("⚠️ Graph file appears corrupted, starting fresh graph.")
            graph = nx.Graph()

    # Rebuild all_interest_phrases from profiles
    for p in all_profiles:
        for phrase in p.get("interest_phrases", []):
            if phrase:
                all_interest_phrases[phrase].append(p["user_id"])

    return all_profiles, visited_depths, graph, all_interest_phrases

# === MAIN CRAWLING FUNCTION ===
def crawl_bfs_resume(driver, queue, all_profiles, visited_depths, force=False):
    """Main BFS crawling function with resume capability"""
    start_time = time.time()
    total_scraped = 0
    last_insert_check = time.time()

    # Create visited_ids set
    visited_ids = set()
    for p in all_profiles:
        if isinstance(p, dict):
            visited_ids.add(p.get("user_id"))
        elif isinstance(p, tuple):
            visited_ids.add(p[0])

    # Handle force mode
    if force:
        queued_user_ids = set(item[0] for item in queue)
        for uid in queued_user_ids:
            if uid in visited_depths:
                print(f"⚠️ Forcing re-process of user {uid}")
                visited_depths.pop(uid)

    while queue:
        # Check for new insertions periodically
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
                            if user_id not in visited_depths and not any(user_id == q[0] for q in queue):
                                new_items.append((user_id, depth, parent))
                        except Exception as e:
                            print(f"⚠️ Invalid line in insert file: {line} — {e}")
                    
                    if new_items:
                        print(f"📥 Inserting {len(new_items)} new users into the queue")
                        for item in reversed(new_items):
                            queue.appendleft(item)
                except Exception as e:
                    print(f"❌ Error reading from insert file: {e}")
            last_insert_check = time.time()

        user_id, depth, parent_id = queue.popleft()
        depth = int(depth)

        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > MAX_CRAWL_SECONDS:
            print(f"🛑 Max crawl time {MAX_CRAWL_SECONDS}s reached, ending crawl.")
            break

        # Skip if already visited
        if not force and user_id in visited_depths:
            print(f"Skipping already visited user {user_id}")
            continue

        print(f"🔎 Crawling {user_id} at depth {depth}")
        try:
            # Extract profile
            profile = extract_profile(driver, user_id, depth)
            visited_depths[user_id] = depth

            # Run fuzzy matching
            run_fuzzy_matching_single(profile)

            # Get Wikipedia info
            wiki_info = get_author_wikipedia_info(profile.get("name", ""))
            profile.update({
                "wiki_birth_name": wiki_info.get("birth_name", ""),
                "wiki_name": wiki_info.get("name", ""),
                "wiki_birth_date": wiki_info.get("birth_date", ""),
                "wiki_birth_place": wiki_info.get("birth_place", ""),
                "wiki_death_date": wiki_info.get("death_date", ""),
                "wiki_death_place": wiki_info.get("death_place", ""),
                "wiki_fields": wiki_info.get("fields", ""),
                "wiki_work_institution": wiki_info.get("work_institution", ""),
                "wiki_alma_mater": wiki_info.get("alma_mater", ""),
                "wiki_notable_students": wiki_info.get("notable_students", ""),
                "wiki_thesis_title": wiki_info.get("thesis_title", ""),
                "wiki_thesis_year": wiki_info.get("thesis_year", ""),
                "wiki_thesis_url": wiki_info.get("thesis_url", ""),
                "wiki_known_for": wiki_info.get("known_for", ""),
                "wiki_awards": wiki_info.get("awards", ""),
                "wiki_deceased": wiki_info.get("deceased", False),
                "wiki_wiki_summary": wiki_info.get("wiki_summary", ""),
                "wiki_is_researcher_ml": wiki_info.get("is_researcher_ml", False),
                "wiki_matched_title": wiki_info.get("matched_title", None)
            })

            # Add profile to collection
            if isinstance(profile, dict):
                all_profiles.append(profile)
            else:
                print(f"⚠️ Skipping non-dict profile: {profile}")
            total_scraped += 1

            print(f"✅ {profile['name']} | h-index: {profile['h_index_all']} | {profile['country']} | depth {depth}")

            # Enqueue co-authors (limit 20)
            for co_id in profile.get("coauthors", [])[:20]:
                if co_id not in visited_depths:
                    enqueue_user(queue, co_id, depth + 1, user_id)

            save_queue(queue)

            # Save progress periodically
            if total_scraped % SAVE_EVERY == 0:
                save_progress(all_profiles)
                print(f"💾 Saved progress after {total_scraped} profiles.")
                
        except Exception as e:
            print(f"❌ Error scraping {user_id}: {e}")

    print(f"✅ BFS crawl finished with {total_scraped} new profiles.")
    save_progress(all_profiles)
    save_queue(queue)

# === MAIN EXECUTION ===
def main():
    """Main execution function"""
    # Initialize components
    model = SentenceTransformer("all-MiniLM-L6-v2")
    driver = get_driver()
    progress_bar = tqdm(total=0, desc="Crawling profiles", dynamic_ncols=True)

    # Ensure CSV is properly initialized
    ensure_progress_csv(PROGRESS_CSV)

    # Load existing progress
    all_profiles, visited_depths, graph, all_interest_phrases = load_progress()

    # Initialize queue
    queue = deque()

    if all_profiles:
        # Resume from existing profiles
        last_users = [p["user_id"] for p in all_profiles[-10:]]
        max_depth = max(visited_depths.get(uid, 0) for uid in last_users)
        print(f"🔍 Fetching co-authors of last 10 users: {last_users}")
    
        coauthors = set()
        for user_id in last_users:
            try:
                new_coauthors = get_coauthors_from_profile(driver, user_id)
                for co in new_coauthors:
                    if co not in visited_depths:
                        coauthors.add(co)
            except Exception as e:
                print(f"❌ Failed to fetch co-authors for {user_id}: {e}")
    
        if coauthors:
            for co in coauthors:
                queue.append((co, max_depth + 1, None))
            print(f"✅ Queue initialized with {len(queue)} new co-authors at depth {max_depth + 1}.")
        else:
            print(f"⚠️ No new co-authors found, starting from seed user {SEED_USER_ID}")
            queue.append((SEED_USER_ID, 0, None))
    else:
        # Start fresh
        print(f"🔄 Starting fresh from seed {SEED_USER_ID}")
        queue.append((SEED_USER_ID, 0, None))

    try:
        # Run main crawling
        crawl_bfs_resume(driver, queue, all_profiles, visited_depths, force=True)

        # Perform phrase clustering
        all_interest_phrases_list = [phrase for p in all_profiles for phrase in p.get("interest_phrases", [])]
        unique_phrases = list(set(all_interest_phrases_list))
        print(f"📊 Clustering {len(unique_phrases)} phrases...")

        if len(unique_phrases) >= 2:
            embeddings = model.encode(unique_phrases)
            n_clusters = min(NUM_PHRASE_CLUSTERS, len(unique_phrases))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
            phrase_to_cluster = {phrase: int(label) for phrase, label in zip(unique_phrases, kmeans.labels_)}

            # Update profiles with cluster information
            for p in all_profiles:
                clusters = sorted(set(phrase_to_cluster.get(ph) for ph in p.get("interest_phrases", []) if ph in phrase_to_cluster))
                p["topic_clusters"] = clusters

        # Final save
        save_progress(all_profiles)
        print("✅ Profiles saved with clustering complete!")

    finally:
        progress_bar.close()
        driver.quit()

if __name__ == "__main__":
    main()
