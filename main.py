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


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === DEVICE & MODEL SETUP ===
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
PROGRESS_CSV = "scholar_profiles_progressssss.csv"
GRAPH_GRAPHML = "coauthor_network_progressssss.graphml"
QUEUE_FILE = "queue.txt"
MAX_CRAWL_SECONDS = 3600000
FUZZY_CACHE_PATH = "fuzzy_match_cache.json"
ICLR_PARQUET_PATH = 'iclr_2020_2025_combined_data.parquet'
NEURIPS_PARQUET_PATH = 'neurips_2020_2024_combined_data.parquet'
FUZZY_RUN_INTERVAL = 5
phrase_to_topics = {}

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
        print("Classify researcher from summary", time.time() - start)
        return top_label in [label.lower() for label in researcher_labels]
    except Exception:
        return False

def tag_interest_phrases(phrases):
    """Tag interest phrases with research topics"""
    start = time.time()
    tags = set()
    for phrase in phrases:
        if phrase not in phrase_to_topics:
            try:
                result = classifier(phrase, topic_labels)
                top_labels = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.3]
                phrase_to_topics[phrase] = top_labels
            except Exception:
                phrase_to_topics[phrase] = []
        tags.update(phrase_to_topics[phrase])
        print("tag_interest_phrases", time.time() - start)
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
    print("clean wiki markup", time.time() - start)
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
    print("fuzzy wiki search", time.time() - start)
    return best_match if best_score >= threshold else None

def get_wikipedia_summary(page_title):
    """Get Wikipedia page summary"""
    S = requests.Session()
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"
    S.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; WikiInfoBot/1.0)'})
    response = S.get(url)
    return data.get("extract", "") if response.status_code == 200 and (data := response.json()) else ""

def get_selected_infobox_fields(page_title, fields_to_extract):
    """Extract specific fields from Wikipedia infobox"""
    S = requests.Session()
    S.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; WikiInfoBot/1.0)'})
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {"action": "query", "format": "json", "titles": page_title,
              "prop": "revisions", "rvprop": "content", "rvslots": "main"}
    
    for _ in range(3):  # Retry logic
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

def get_author_wikipedia_info(author_name):
    """Get comprehensive Wikipedia info for an author"""
    fields = ["birth_name", "name", "birth_date", "birth_place", "death_date", "death_place",
              "fields", "work_institution", "alma_mater", "notable_students", "thesis_title",
              "thesis_year", "thesis_url", "known_for", "awards"]

    matched_title = fuzzy_wikipedia_search(author_name)
    if matched_title is None:
        info = {k: "" for k in fields}
        info.update({"deceased": False, "wiki_summary": "", "is_researcher_ml": False, "matched_title": None})
        return info

    info, matched_title = get_selected_infobox_fields(matched_title, fields)
    if info is None:
        info = {k: "" for k in fields}
        info["deceased"] = False
    else:
        info["deceased"] = bool(info.get("death_date"))
    
    summary = get_wikipedia_summary(matched_title)
    info.update({
        "wiki_summary": summary,
        "is_researcher_ml": classify_researcher_from_summary(summary),
        "matched_title": matched_title
    })
    
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
    """Initialize headless Chrome driver"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)

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
    """Extract complete profile from Google Scholar"""
    url = f"https://scholar.google.com/citations?hl=en&user={user_id}"
    driver.get(url)
    try:
        WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.ID, "gsc_prf_in"))
        )
    except Exception as e:
        print(f"⚠️ Wait timeout for {url} — {e}")
        time.sleep(2)  # Fallback sleep if Google Scholar is slow
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
        if homepage_link and homepage_link.get_text(strip=True).lower() == "homepage":
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
        if "user=" in href:
            co_id = href.split("user=")[1].split("&")[0]
            if co_id != user_id:
                coauthors.append(co_id)

    # Add parent to coauthors if exists
    if parent_id and parent_id != user_id and parent_id not in coauthors:
        coauthors.append(parent_id)

    print("extract profile", time.time() - start)

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
            df = pd.read_csv(path)
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
    authors_lower = [a.lower() for a in authors]

    profile_name = profile.get("name", "").lower()
    if not profile_name:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Institution"] = ""
        return

    match, score = process.extractOne(profile_name, authors_lower, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        match_idx = authors_lower.index(match)
        matched_author = authors[match_idx]
        if inst_col in df.columns:
            inst_values = df[df[name_col] == matched_author][inst_col].dropna()
            matched_institution = inst_values.iloc[0] if not inst_values.empty else ""
        else:
            matched_institution = ""
        profile[f"Participated_in_{conf_name}"] = True
        profile[f"{conf_name}_Institution"] = matched_institution
    else:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Institution"] = ""

def run_fuzzy_matching_single(profile):
    """Run fuzzy matching for a single profile"""
    if profile.get("Fuzzy_Matched"):
        return

    # ICLR matching
    if os.path.exists(ICLR_PARQUET_PATH):
        iclr_df = pd.read_parquet(ICLR_PARQUET_PATH)
        fuzzy_match_conference_participation(profile, "ICLR", iclr_df)

    # NeurIPS matching
    if os.path.exists(NEURIPS_PARQUET_PATH):
        neurips_df = pd.read_parquet(NEURIPS_PARQUET_PATH)
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
            profiles_df = pd.read_csv(PROGRESS_CSV)
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
        if "user=" in href:
            co_id = href.split("user=")[1].split("&")[0]
            coauthors.append(co_id)
    return coauthors

def enrich_profile(profile):
    try:
        # Tag interests
        if "interest_phrases" in profile and profile["interest_phrases"]:
            profile["topic_tags"] = tag_interest_phrases(profile["interest_phrases"])
        else:
            profile["topic_tags"] = []

        # Fuzzy matching
        run_fuzzy_matching_single(profile)

        # Wikipedia info
        wiki_info = get_author_wikipedia_info(profile.get("name", ""))
        profile.update({
            f"wiki_{k}": v for k, v in wiki_info.items()
        })

        return profile
    except Exception as e:
        print(f"❌ Enrichment failed for {profile.get('user_id')}: {e}")
        return profile  # Return even if partial

# === MAIN CRAWLING FUNCTION ===
def crawl_bfs_resume(driver, queue, all_profiles, visited_depths, force=False):
    """Main BFS crawling function with batched multiprocessing enrichment"""
    start_time = time.time()
    total_scraped = 0
    INSERT_FILE = "queue_insert.jsonl"
    INSERT_CHECK_INTERVAL = 30
    last_insert_check = time.time()

    visited_ids = {p.get("user_id") for p in all_profiles if isinstance(p, dict)}

    if force:
        queued_user_ids = set(item[0] for item in queue)
        for uid in queued_user_ids:
            visited_depths.pop(uid, None)

    profile_batch = []
    BATCH_SIZE = 100

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
                            if user_id not in visited_depths and not any(user_id == q[0] for q in queue):
                                new_items.append((user_id, depth, parent))
                        except Exception:
                            continue
                    
                    if new_items:
                        for item in reversed(new_items):
                            queue.appendleft(item)
                except Exception:
                    pass
            last_insert_check = time.time()

        user_id, depth, parent_id = queue.popleft()
        depth = int(depth)

        if time.time() - start_time > MAX_CRAWL_SECONDS:
            break

        if not force and user_id in visited_depths:
            continue

        try:
            profile = extract_profile(driver, user_id, depth, parent_id)
            visited_depths[user_id] = depth
            profile_batch.append(profile)
            total_scraped += 1

            # Enqueue coauthors
            for co_id in profile.get("coauthors", [])[:20]:
                if co_id not in visited_depths and co_id != user_id:
                    enqueue_user(queue, co_id, depth + 1, user_id)

            # Process batch if size reached
            if len(profile_batch) >= BATCH_SIZE:
                with Pool(cpu_count()-4) as pool:
                    enriched_profiles = pool.map(enrich_profile, profile_batch)
                all_profiles.extend(enriched_profiles)
                profile_batch = []

                print(f"✅ Processed {total_scraped} profiles so far")
                
                save_queue(queue)
                save_progress(all_profiles)

        except Exception as e:
            print(f"❌ Error scraping {user_id}: {e}")

    # Flush leftover profiles after queue emptied
    if profile_batch:
        with Pool(cpu_count()) as pool:
            enriched_profiles = pool.map(enrich_profile, profile_batch)
        all_profiles.extend(enriched_profiles)
        profile_batch = []
        save_progress(all_profiles)

    print("Crawl BFS", time.time() - start_time)

    save_queue(queue)
    save_progress(all_profiles)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Initialize globals
    phrase_to_topics = {}
    driver = get_driver()
    progress_bar = tqdm(total=0, desc="Crawling profiles", dynamic_ncols=True)

    # Load progress
    all_profiles, visited_depths, graph, all_interest_phrases = load_progress()
    ensure_progress_csv(PROGRESS_CSV)

    # === Initialize queue with multi-layer fallback ===
    if os.path.exists(QUEUE_FILE):
        queue = load_queue()
        print(len(queue))
    else:
        queue = deque()
        if all_profiles:
            last_users = [p["user_id"] for p in all_profiles[-10:] if p.get("user_id")]
            max_depth = max(visited_depths.get(uid, 0) for uid in last_users)
    
            coauthors = set()
            for user_id in last_users:
                try:
                    new_coauthors = get_coauthors_from_profile(driver, user_id)
                    for co in new_coauthors:
                        if co not in visited_depths:
                            coauthors.add(co)
                except Exception:
                    continue
    
            if coauthors:
                for co in coauthors:
                    queue.append((co, max_depth + 1, None))
            else:
                queue.append((SEED_USER_ID, 0, None))
        else:
            queue.append((SEED_USER_ID, 0, None))
    try:
        crawl_bfs_resume(driver, queue, all_profiles, visited_depths, force=True)
        print("✅ Crawling complete!")
    finally:
        progress_bar.close()
        driver.quit()
