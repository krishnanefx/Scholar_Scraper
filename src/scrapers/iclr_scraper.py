#!/usr/bin/env python3
"""
Enhanced ICLR Scraper with Topic Classification

This is an improved version that includes more sophisticated topic classification
and better integration with the existing codebase structure.
"""

import requests
import pandas as pd
import json
import time
import re
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedICLRScraper:
    """Enhanced ICLR Conference Data Scraper with better topic classification"""
    
    # ICLR URLs to scrape
    ICLR_URLS = {
        2020: "https://iclr.cc/static/virtual/data/iclr-2020-orals-posters.json",
        2021: "https://iclr.cc/static/virtual/data/iclr-2021-orals-posters.json", 
        2022: "https://iclr.cc/static/virtual/data/iclr-2022-orals-posters.json",
        2023: "https://iclr.cc/static/virtual/data/iclr-2023-orals-posters.json",
        2024: "https://iclr.cc/static/virtual/data/iclr-2024-orals-posters.json",
        2025: "https://iclr.cc/static/virtual/data/iclr-2025-orals-posters.json"
    }
    
    # Enhanced topic classification patterns
    TOPIC_PATTERNS = {
        'Computer Vision': [
            'vision', 'image', 'visual', 'cnn', 'convolutional', 'detection', 'segmentation', 
            'classification', 'recognition', 'object', 'pixel', 'convnet', 'resnet', 'vgg',
            'imagenet', 'cifar', 'opencv', 'video', 'frame', 'optical'
        ],
        'Natural Language Processing': [
            'nlp', 'language', 'text', 'linguistic', 'translation', 'bert', 'gpt', 'transformer',
            'attention', 'embedding', 'word', 'sentence', 'parsing', 'semantic', 'syntax',
            'machine translation', 'sentiment', 'qa', 'question answering', 'dialog'
        ],
        'Reinforcement Learning': [
            'reinforcement', 'rl', 'policy', 'reward', 'agent', 'environment', 'action',
            'q-learning', 'actor-critic', 'monte carlo', 'temporal difference', 'exploration',
            'exploitation', 'markov', 'mdp', 'bandit', 'game'
        ],
        'Deep Learning': [
            'neural network', 'deep', 'backpropagation', 'gradient', 'layer', 'activation',
            'dropout', 'batch norm', 'weight', 'bias', 'loss function', 'optimizer',
            'sgd', 'adam', 'rmsprop', 'learning rate', 'epoch', 'batch'
        ],
        'Graph Neural Networks': [
            'graph', 'node', 'edge', 'gnn', 'gcn', 'graph neural', 'network', 'adjacency',
            'vertex', 'topology', 'social network', 'knowledge graph', 'graph convolution'
        ],
        'Optimization': [
            'optimization', 'minimize', 'maximize', 'convex', 'non-convex', 'constraint',
            'linear programming', 'quadratic', 'gradient descent', 'newton', 'quasi-newton',
            'convergence', 'global minimum', 'local minimum'
        ],
        'Machine Learning': [
            'supervised', 'unsupervised', 'semi-supervised', 'classification', 'regression',
            'clustering', 'dimensionality reduction', 'feature selection', 'cross-validation',
            'overfitting', 'underfitting', 'bias', 'variance', 'ensemble', 'bagging', 'boosting'
        ],
        'Generative Models': [
            'generative', 'gan', 'vae', 'variational', 'autoencoder', 'decoder', 'encoder',
            'latent', 'sampling', 'distribution', 'likelihood', 'generative adversarial'
        ]
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
    def classify_topic(self, title: str, abstract: str) -> Tuple[str, str]:
        """Enhanced topic classification using keyword matching and patterns"""
        text = f"{title} {abstract}".lower()
        
        # Clean the text
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        topic_scores = {}
        
        # Calculate scores for each topic
        for topic, patterns in self.TOPIC_PATTERNS.items():
            score = 0
            for pattern in patterns:
                # Count occurrences of each pattern
                count = len(re.findall(r'\b' + re.escape(pattern) + r'\b', text))
                score += count
            topic_scores[topic] = score
        
        # Find the topic with highest score
        if topic_scores and max(topic_scores.values()) > 0:
            main_topic = max(topic_scores.keys(), key=lambda k: topic_scores[k])
        else:
            main_topic = 'Other'
        
        # Determine subtopic based on specific keywords
        if 'oral' in title.lower() or 'oral' in abstract.lower():
            subtopic = 'Oral Presentation'
        elif 'poster' in title.lower() or 'poster' in abstract.lower():
            subtopic = 'Poster'
        elif any(word in text for word in ['workshop', 'demo', 'demonstration']):
            subtopic = 'Workshop'
        else:
            subtopic = 'Paper'
            
        return main_topic, subtopic
    
    def extract_keywords(self, title: str, abstract: str) -> str:
        """Extract relevant keywords from title and abstract"""
        text = f"{title} {abstract}".lower()
        
        # Common ML/AI keywords to extract
        keyword_patterns = [
            r'\b(?:neural|network|deep|learning|machine|artificial|intelligence)\b',
            r'\b(?:algorithm|model|training|testing|validation)\b', 
            r'\b(?:classification|regression|clustering|optimization)\b',
            r'\b(?:supervised|unsupervised|reinforcement|semi-supervised)\b',
            r'\b(?:cnn|rnn|lstm|gru|transformer|attention|bert|gpt)\b',
            r'\b(?:gan|vae|autoencoder|generative|discriminative)\b'
        ]
        
        keywords = set()
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text)
            keywords.update(matches)
            
        return ', '.join(sorted(keywords)[:10])  # Limit to top 10 keywords
    
    def fetch_and_process_data(self) -> pd.DataFrame:
        """Fetch and process all ICLR data"""
        all_records = []
        
        for year, url in self.ICLR_URLS.items():
            try:
                logger.info(f"Processing ICLR {year}...")
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                papers = data.get('results', [])
                
                logger.info(f"Found {len(papers)} papers for {year}")
                
                for paper in papers:
                    records = self._process_paper(paper, year)
                    all_records.extend(records)
                
                time.sleep(1)  # Be respectful to the server
                
            except Exception as e:
                logger.error(f"Error processing ICLR {year}: {e}")
                continue
        
        return pd.DataFrame(all_records)
    
    def _process_paper(self, paper: Dict[str, Any], year: int) -> List[Dict[str, Any]]:
        """Process a single paper and return records for each author"""
        title = (paper.get('name') or '').strip()
        abstract = (paper.get('abstract') or '').strip()
        
        # Enhanced topic classification
        main_topic, subtopic = self.classify_topic(title, abstract)
        keywords = self.extract_keywords(title, abstract)
        
        # Extract paper URL
        paper_url = (paper.get('sourceurl') or 
                    paper.get('paper_pdf_url') or 
                    paper.get('url') or '').strip()
        
        authors = paper.get('authors', [])
        if not authors:
            authors = [{}]  # Create empty author record
        
        records = []
        for author in authors:
            author_name = (author.get('fullname') or '').strip()
            institution = (author.get('institution') or '').strip()
            
            record = {
                'Year': year,
                'Title': title,
                'Author': author_name,
                'Institution': institution,
                'Abstract': abstract,
                'Main Topic': main_topic,
                'Subtopic': subtopic,
                'Keywords': keywords,
                'Paper URL': paper_url,
                # Wikipedia fields (empty for now - can be filled later)
                'birth_name': '',
                'name': author_name,
                'birth_date': '',
                'birth_place': '',
                'death_date': '',
                'death_place': '',
                'fields': '',
                'work_institution': institution,
                'alma_mater': '',
                'notable_students': '',
                'thesis_title': '',
                'thesis_year': '',
                'thesis_url': '',
                'known_for': '',
                'awards': '',
                'deceased': False,
                'wiki_summary': '',
                'is_researcher_ml': False
            }
            records.append(record)
        
        return records
    
    def save_results(self, df: pd.DataFrame, output_prefix: str = "iclr_enhanced"):
        """Save results in multiple formats"""
        
        # Ensure output directory exists
        output_dir = Path("../../data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = output_dir / f"{output_prefix}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")
        
        # Save as Parquet
        parquet_path = output_dir / f"{output_prefix}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved Parquet: {parquet_path}")
        
        # Print summary
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Unique authors: {df['Author'].nunique()}")
        logger.info(f"Years: {df['Year'].min()} - {df['Year'].max()}")
        
        print("\nTopic Distribution:")
        topic_dist = df['Main Topic'].value_counts()
        for topic, count in topic_dist.items():
            print(f"  {topic}: {count}")
        
        print(f"\nSample of processed data:")
        print(df[['Year', 'Title', 'Author', 'Institution', 'Main Topic']].head())

def main():
    """Main execution function"""
    logger.info("Starting enhanced ICLR data collection...")
    
    scraper = EnhancedICLRScraper()
    
    # Fetch and process data
    df = scraper.fetch_and_process_data()
    
    if df.empty:
        logger.error("No data collected!")
        return
    
    # Save results
    scraper.save_results(df)
    
    logger.info("Enhanced ICLR scraping completed successfully!")

if __name__ == "__main__":
    main()
