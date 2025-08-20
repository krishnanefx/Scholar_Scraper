#!/usr/bin/env python3
"""
NeurIPS Conference Data Scraper

Clean scraper to fetch NeurIPS data from 2020-2024 and save as parquet.
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

class NeurIPSScraper:
    """NeurIPS Conference Data Scraper"""
    
    # NeurIPS URLs to scrape
    NEURIPS_URLS = {
        2020: "https://neurips.cc/static/virtual/data/neurips-2020-orals-posters.json",
        2021: "https://neurips.cc/static/virtual/data/neurips-2021-orals-posters.json",
        2022: "https://neurips.cc/static/virtual/data/neurips-2022-orals-posters.json",
        2023: "https://neurips.cc/static/virtual/data/neurips-2023-orals-posters.json",
        2024: "https://neurips.cc/static/virtual/data/neurips-2024-orals-posters.json"
    }
    
    # Topic classification patterns
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
        ],
        'Theory': [
            'theoretical', 'theory', 'proof', 'lemma', 'theorem', 'complexity', 'bounds',
            'convergence', 'analysis', 'mathematical', 'statistics', 'probability'
        ]
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
    def classify_topic(self, title: str, abstract: str) -> str:
        """Topic classification using keyword matching"""
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
            
        return main_topic
    
    def fetch_and_process_data(self) -> pd.DataFrame:
        """Fetch and process all NeurIPS data"""
        all_records = []
        
        for year, url in self.NEURIPS_URLS.items():
            try:
                logger.info(f"Processing NeurIPS {year}...")
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Handle different JSON structures
                if isinstance(data, list):
                    papers = data
                elif isinstance(data, dict):
                    papers = data.get('results', data.get('papers', []))
                else:
                    logger.error(f"Unexpected data format for {year}")
                    continue
                
                logger.info(f"Found {len(papers)} papers for {year}")
                
                for paper in papers:
                    records = self._process_paper(paper, year)
                    all_records.extend(records)
                
                time.sleep(1)  # Be respectful to the server
                
            except Exception as e:
                logger.error(f"Error processing NeurIPS {year}: {e}")
                continue
        
        return pd.DataFrame(all_records)
    
    def _process_paper(self, paper: Dict[str, Any], year: int) -> List[Dict[str, Any]]:
        """Process a single paper and return records for each author"""
        # Handle different possible field names
        title = (paper.get('title') or 
                paper.get('name') or 
                paper.get('paper_title') or '').strip()
        
        abstract = (paper.get('abstract') or 
                   paper.get('description') or '').strip()
        
        # Topic classification
        main_topic = self.classify_topic(title, abstract)
        
        # Extract paper URL
        paper_url = (paper.get('paper_url') or 
                    paper.get('url') or 
                    paper.get('sourceurl') or 
                    paper.get('pdf_url') or '').strip()
        
        # Handle authors - different possible structures
        authors = []
        if 'authors' in paper:
            authors = paper['authors']
        elif 'author' in paper:
            authors = [paper['author']] if isinstance(paper['author'], dict) else paper['author']
        elif 'contributors' in paper:
            authors = paper['contributors']
        
        if not authors:
            authors = [{}]  # Create empty author record
        
        records = []
        for author in authors:
            if isinstance(author, str):
                author_name = author.strip()
                institution = ''
            else:
                author_name = (author.get('name') or 
                             author.get('fullname') or 
                             author.get('author') or '').strip()
                institution = (author.get('institution') or 
                             author.get('affiliation') or 
                             author.get('org') or '').strip()
            
            record = {
                'Year': year,
                'Title': title,
                'Author': author_name,
                'Institution': institution,
                'Abstract': abstract,
                'Main Topic': main_topic,
                'Paper URL': paper_url,
                # Standard fields for compatibility with existing models
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
    
    def save_results(self, df: pd.DataFrame):
        """Save results as parquet"""
        
        # Ensure output directory exists
        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
        parquet_path = output_dir / "neurips_2020_2024_combined_data.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved Parquet: {parquet_path}")
        
        # Print summary
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Unique authors: {df['Author'].nunique()}")
        logger.info(f"Unique papers: {df['Title'].nunique()}")
        logger.info(f"Years: {df['Year'].min()} - {df['Year'].max()}")
        
        print("\nTopic Distribution:")
        topic_dist = df['Main Topic'].value_counts()
        for topic, count in topic_dist.items():
            print(f"  {topic}: {count}")
        
        print(f"\nYear Distribution:")
        year_dist = df['Year'].value_counts().sort_index()
        for year, count in year_dist.items():
            print(f"  {year}: {count}")

def main():
    """Main execution function"""
    logger.info("Starting NeurIPS data collection...")
    
    scraper = NeurIPSScraper()
    
    # Fetch and process data
    df = scraper.fetch_and_process_data()
    
    if df.empty:
        logger.error("No data collected!")
        return
    
    # Save results
    scraper.save_results(df)
    
    logger.info("NeurIPS scraping completed successfully!")

if __name__ == "__main__":
    main()
