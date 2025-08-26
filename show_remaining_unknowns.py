#!/usr/bin/env python3
"""
Final Interactive Domain Mapper - Shows remaining unknown domains
"""

import pandas as pd
import re
import tldextract
import os
import json
from collections import Counter

def load_current_mappings():
    """Load current mappings from JSON file"""
    if os.path.exists("domain_mappings.json"):
        with open("domain_mappings.json", 'r') as f:
            mappings = json.load(f)
            return mappings.get('countries', {}), mappings.get('institutions', {})
    return {}, {}

def extract_domain_from_email(email_field):
    """Extract domain from email field"""
    if pd.isna(email_field) or email_field == "":
        return None
    match = re.search(r"Verified email at ([^\s]+?)(?:\s*-\s*Homepage)?$", str(email_field))
    if not match:
        return None
    return match.group(1).lower().strip()

def infer_institution_from_domain(domain, domain_to_institution):
    """Infer institution from domain"""
    if not domain:
        return "Unknown"
    for known_domain in domain_to_institution:
        if domain.endswith(known_domain):
            return domain_to_institution[known_domain]
    return "Unknown"

def get_unknown_institution_domains(df, domain_to_institution, top_n=100):
    """Get top N domains that don't have institution mappings"""
    unknown_domains = []
    
    for _, row in df.iterrows():
        if row['institution'] == 'Unknown':
            domain = extract_domain_from_email(row['email'])
            if domain:
                institution = infer_institution_from_domain(domain, domain_to_institution)
                if institution == "Unknown":
                    unknown_domains.append(domain)
    
    # Count occurrences
    domain_counts = Counter(unknown_domains)
    
    # Get top N unknown domains
    top_domains = []
    for domain, count in domain_counts.most_common(top_n):
        top_domains.append({
            'domain': domain,
            'count': count
        })
    
    return top_domains

def show_statistics_and_unknown_domains():
    """Show current statistics and unknown domains"""
    
    # Load data
    csv_path = "data/processed/scholar_profiles.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Load current mappings
    suffix_country_map, domain_to_institution = load_current_mappings()
    
    # Calculate statistics
    total_records = len(df)
    unknown_countries = len(df[df['country'] == 'Unknown'])
    unknown_institutions = len(df[df['institution'] == 'Unknown'])
    
    print("🎯 Current Scholar Profiles Statistics")
    print("=" * 50)
    print(f"Total records: {total_records:,}")
    print(f"Unknown countries: {unknown_countries:,} ({unknown_countries/total_records*100:.1f}%)")
    print(f"Unknown institutions: {unknown_institutions:,} ({unknown_institutions/total_records*100:.1f}%)")
    print(f"Mapped institutions: {total_records - unknown_institutions:,} ({(total_records - unknown_institutions)/total_records*100:.1f}%)")
    
    if unknown_institutions == 0:
        print("\n🎉 Congratulations! All institutions have been mapped!")
        return
    
    # Get unknown domains
    unknown_domains = get_unknown_institution_domains(df, domain_to_institution, 100)
    
    print(f"\n📊 Top {min(100, len(unknown_domains))} domains with unknown institutions:")
    print("=" * 70)
    print(f"{'Rank':<4} {'Domain':<35} {'Count':<8} {'Sample Records'}")
    print("-" * 70)
    
    for i, domain_info in enumerate(unknown_domains[:50], 1):  # Show top 50
        domain = domain_info['domain']
        count = domain_info['count']
        
        # Get some sample names from this domain
        sample_records = df[
            (df['institution'] == 'Unknown') & 
            (df['email'].str.contains(f"at {re.escape(domain)}", na=False, case=False))
        ]['name'].head(3).tolist()
        
        sample_text = "; ".join(sample_records[:2])
        if len(sample_text) > 30:
            sample_text = sample_text[:27] + "..."
        
        print(f"{i:<4} {domain:<35} {count:<8} {sample_text}")
    
    if len(unknown_domains) > 50:
        print(f"\n... and {len(unknown_domains) - 50} more domains")
    
    print(f"\n💡 To continue mapping:")
    print("1. Add mappings to domain_mappings.json manually, or")
    print("2. Use the interactive_domain_mapper.py script")
    print("3. Run fix_mappings.py again to apply new mappings")
    
    # Show some examples of what to add
    if unknown_domains:
        print(f"\n📝 Example JSON additions for domain_mappings.json:")
        print("Add to the 'institutions' section:")
        for domain_info in unknown_domains[:5]:
            domain = domain_info['domain']
            print(f'    "{domain}": "Your Institution Name Here",')

if __name__ == "__main__":
    show_statistics_and_unknown_domains()
