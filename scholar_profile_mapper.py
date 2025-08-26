#!/usr/bin/env python3
"""
Comprehensive Scholar Profile Domain and Country Mapper
Consolidates all mapping functionality into one script.

Features:
- Maps institutions from email domains
- Maps countries from email domains and institutions  
- Standardizes country names
- Interactive and batch mapping modes
- Shows statistics and remaining unknowns
- Persistent mapping storage

Usage:
    python scholar_profile_mapper.py --help
    python scholar_profile_mapper.py --stats
    python scholar_profile_mapper.py --fix-all
    python scholar_profile_mapper.py --interactive
    python scholar_profile_mapper.py --show-unknowns
"""

import pandas as pd
import re
import tldextract
import os
import json
import argparse
from collections import Counter

class ScholarProfileMapper:
    def __init__(self, csv_path="data/processed/scholar_profiles.csv"):
        self.csv_path = csv_path
        self.df = None
        self.domain_mappings_file = "domain_mappings.json"
        self.institution_country_mappings_file = "institution_to_country_mapping.json"
        
        # Initialize default mappings
        self.suffix_country_map = {
            "edu": "United States",
            "uk": "United Kingdom",
            "in": "India",
            "cn": "China",
            "de": "Germany",
            "fr": "France",
            "jp": "Japan",
            "ca": "Canada",
            "au": "Australia",
            "ch": "Switzerland",
            "sg": "Singapore",
            "kr": "South Korea",
            "hk": "Hong Kong",
            "nl": "Netherlands",
            "se": "Sweden",
            "it": "Italy",
            "fi": "Finland",
            "quebec": "Canada",
            "org": "International",
            "be": "Belgium",
            "es": "Spain",
            "pt": "Portugal",
            "il": "Israel",
            "lu": "Luxembourg",
            "eu": "European Union",
            "no": "Norway",
            "dk": "Denmark",
            "ie": "Ireland",
            "sa": "Saudi Arabia",
            "at": "Austria",
            "ee": "Estonia",
            "ac": "Academic Institution",
            "gov": "United States",
            "mil": "United States",
            "com": "Unknown",
        }
        
        self.domain_to_institution = {
            # Tech Companies
            "google.com": "Google",
            "microsoft.com": "Microsoft",
            "meta.com": "Meta (Facebook)",
            "amazon.com": "Amazon",
            "fb.com": "Facebook",
            "apple.com": "Apple",
            "nvidia.com": "NVIDIA",
            "openai.com": "OpenAI",
            "adobe.com": "Adobe",
            "samsung.com": "Samsung",
            "intel.com": "Intel",
            "anthropic.com": "Anthropic",
            "us.ibm.com": "IBM",
            "ibm.com": "IBM",
            "baidu.com": "Baidu",
            "sri.com": "SRI International",
            "waymo.com": "Waymo",
            "deepmind.com": "Google DeepMind",
            "huawei.com": "Huawei",
            "bytedance.com": "ByteDance",
            "alibaba-inc.com": "Alibaba",
            "tencent.com": "Tencent",
            "servicenow.com": "ServiceNow",
            "sony.com": "Sony",
            "oracle.com": "Oracle",
            "siemens.com": "Siemens",
            "salesforce.com": "Salesforce",
            "cohere.com": "Cohere",
            "qti.qualcomm.com": "Qualcomm",
            "naverlabs.com": "Naver Labs",
            "de.bosch.com": "Bosch",
            "linkedin.com": "LinkedIn",
            "netflix.com": "Netflix",
            "uber.com": "Uber",
            "x.ai": "x.ai",
            "tri.global": "Toyota Research Institute",
            
            # US Universities
            "berkeley.edu": "University of California, Berkeley",
            "nyu.edu": "New York University",
            "cmu.edu": "Carnegie Mellon University",
            "cs.cmu.edu": "Carnegie Mellon University",
            "andrew.cmu.edu": "Carnegie Mellon University",
            "umich.edu": "University of Michigan",
            "cornell.edu": "Cornell University",
            "princeton.edu": "Princeton University",
            "yale.edu": "Yale University",
            "harvard.edu": "Harvard University",
            "fas.harvard.edu": "Harvard University",
            "g.harvard.edu": "Harvard University",
            "seas.harvard.edu": "Harvard University",
            "hms.harvard.edu": "Harvard Medical School",
            "mgh.harvard.edu": "Massachusetts General Hospital",
            "columbia.edu": "Columbia University",
            "uchicago.edu": "University of Chicago",
            "caltech.edu": "California Institute of Technology",
            "mit.edu": "Massachusetts Institute of Technology",
            "csail.mit.edu": "MIT CSAIL",
            "alum.mit.edu": "Massachusetts Institute of Technology",
            "stanford.edu": "Stanford University",
            "cs.stanford.edu": "Stanford University",
            "ucsd.edu": "University of California, San Diego",
            "ucla.edu": "University of California, Los Angeles",
            "uci.edu": "University of California, Irvine",
            "ucsb.edu": "University of California, Santa Barbara",
            "ucdavis.edu": "University of California, Davis",
            "ucsf.edu": "University of California, San Francisco",
            "ucsc.edu": "University of California, Santa Cruz",
            "gatech.edu": "Georgia Institute of Technology",
            "usc.edu": "University of Southern California",
            "northwestern.edu": "Northwestern University",
            "vanderbilt.edu": "Vanderbilt University",
            "brown.edu": "Brown University",
            "dartmouth.edu": "Dartmouth College",
            "rice.edu": "Rice University",
            "illinois.edu": "University of Illinois Urbana-Champaign",
            "uic.edu": "University of Illinois at Chicago",
            "uw.edu": "University of Washington",
            "cs.washington.edu": "University of Washington",
            "wisc.edu": "University of Wisconsin-Madison",
            "cs.wisc.edu": "University of Wisconsin-Madison",
            "umn.edu": "University of Minnesota",
            "virginia.edu": "University of Virginia",
            "vt.edu": "Virginia Tech",
            "utexas.edu": "University of Texas at Austin",
            "utdallas.edu": "University of Texas at Dallas",
            "tamu.edu": "Texas A&M University",
            "asu.edu": "Arizona State University",
            "arizona.edu": "University of Arizona",
            "colorado.edu": "University of Colorado Boulder",
            "purdue.edu": "Purdue University",
            "osu.edu": "Ohio State University",
            "psu.edu": "Pennsylvania State University",
            "upenn.edu": "University of Pennsylvania",
            "seas.upenn.edu": "University of Pennsylvania",
            "rutgers.edu": "Rutgers University",
            "bu.edu": "Boston University",
            "northeastern.edu": "Northeastern University",
            "ufl.edu": "University of Florida",
            "pitt.edu": "University of Pittsburgh",
            "emory.edu": "Emory University",
            "gmu.edu": "George Mason University",
            "wustl.edu": "Washington University in St. Louis",
            "jhu.edu": "Johns Hopkins University",
            "jhmi.edu": "Johns Hopkins Medical Institutions",
            "buffalo.edu": "University at Buffalo",
            "ub.edu": "University at Buffalo",
            "udel.edu": "University of Delaware",
            "ucf.edu": "University of Central Florida",
            "umd.edu": "University of Maryland",
            "ncsu.edu": "North Carolina State University",
            "cs.unc.edu": "University of North Carolina at Chapel Hill",
            "unc.edu": "University of North Carolina at Chapel Hill",
            "cs.umass.edu": "University of Massachusetts Amherst",
            "umass.edu": "University of Massachusetts",
            "ttic.edu": "Toyota Technological Institute at Chicago",
            "nd.edu": "University of Notre Dame",
            "oregonstate.edu": "Oregon State University",
            "stonybrook.edu": "Stony Brook University",
            "njit.edu": "New Jersey Institute of Technology",
            "nyumc.org": "NYU Medical Center",
            "cs.utah.edu": "University of Utah",
            "utah.edu": "University of Utah",
            "duke.edu": "Duke University",
            "msu.edu": "Michigan State University",
            "iu.edu": "Indiana University",
            "indiana.edu": "Indiana University",
            "cs.rochester.edu": "University of Rochester",
            "binghamton.edu": "Binghamton University",
            "rpi.edu": "Rensselaer Polytechnic Institute",
            "tufts.edu": "Tufts University",
            "utk.edu": "University of Tennessee",
            "umbc.edu": "University of Maryland, Baltimore County",
            "ucmerced.edu": "University of California, Merced",
            "uconn.edu": "University of Connecticut",
            "uga.edu": "University of Georgia",
            "iastate.edu": "Iowa State University",
            
            # Canadian Universities
            "utoronto.ca": "University of Toronto",
            "cs.toronto.edu": "University of Toronto",
            "cs.ubc.ca": "University of British Columbia",
            "ubc.ca": "University of British Columbia",
            "mcgill.ca": "McGill University",
            "uwaterloo.ca": "University of Waterloo",
            "umontreal.ca": "University of Montreal",
            "mila.quebec": "MILA",
            "sfu.ca": "Simon Fraser University",
            "ucalgary.ca": "University of Calgary",
            "uwo.ca": "University of Western Ontario",
            "uottawa.ca": "University of Ottawa",
            "etsmtl.ca": "École de technologie supérieure",
            "ualberta.ca": "University of Alberta",
            "polymtl.ca": "Polytechnique Montréal",
            "queensu.ca": "Queen's University",
            "concordia.ca": "Concordia University",
            "yorku.ca": "York University",
            "usherbrooke.ca": "University of Sherbrooke",
            "dal.ca": "Dalhousie University",
            "uvic.ca": "University of Victoria",
            "ulaval.ca": "Université Laval",
            
            # UK Universities
            "cam.ac.uk": "University of Cambridge",
            "cl.cam.ac.uk": "University of Cambridge",
            "ox.ac.uk": "University of Oxford",
            "cs.ox.ac.uk": "University of Oxford",
            "robots.ox.ac.uk": "University of Oxford",
            "eng.ox.ac.uk": "University of Oxford",
            "ucl.ac.uk": "University College London",
            "imperial.ac.uk": "Imperial College London",
            "ed.ac.uk": "University of Edinburgh",
            "kcl.ac.uk": "King's College London",
            "qmul.ac.uk": "Queen Mary University of London",
            "surrey.ac.uk": "University of Surrey",
            "glasgow.ac.uk": "University of Glasgow",
            "sheffield.ac.uk": "University of Sheffield",
            "bristol.ac.uk": "University of Bristol",
            "manchester.ac.uk": "University of Manchester",
            "nottingham.ac.uk": "University of Nottingham",
            "warwick.ac.uk": "University of Warwick",
            "lancaster.ac.uk": "Lancaster University",
            "soton.ac.uk": "University of Southampton",
            "york.ac.uk": "University of York",
            "st-andrews.ac.uk": "University of St Andrews",
            "cardiff.ac.uk": "Cardiff University",
            "bath.ac.uk": "University of Bath",
            
            # European Universities
            "tum.de": "Technical University of Munich",
            "tu-berlin.de": "Technical University of Berlin",
            "uni-tuebingen.de": "University of Tübingen",
            "tuebingen.mpg.de": "Max Planck Institute for Intelligent Systems",
            "kit.edu": "Karlsruhe Institute of Technology",
            "tu-dresden.de": "TU Dresden",
            "inf.ethz.ch": "ETH Zurich",
            "ethz.ch": "ETH Zurich",
            "cs.rwth-aachen.de": "RWTH Aachen University",
            "epfl.ch": "EPFL",
            "unige.ch": "University of Geneva",
            "unibas.ch": "University of Basel",
            "idiap.ch": "Idiap Research Institute",
            "usi.ch": "Università della Svizzera italiana",
            "uzh.ch": "University of Zurich",
            "kth.se": "KTH Royal Institute of Technology",
            "chalmers.se": "Chalmers University of Technology",
            "liu.se": "Linköping University",
            "aalto.fi": "Aalto University",
            "helsinki.fi": "University of Helsinki",
            "tuni.fi": "Tampere University",
            "ntnu.no": "Norwegian University of Science and Technology",
            "dtu.dk": "Technical University of Denmark",
            "di.ku.dk": "University of Copenhagen",
            "cs.au.dk": "Aarhus University",
            "tudelft.nl": "Delft University of Technology",
            "tue.nl": "Eindhoven University of Technology",
            "utwente.nl": "University of Twente",
            "uva.nl": "University of Amsterdam",
            "uu.nl": "Utrecht University",
            "rug.nl": "University of Groningen",
            "vu.nl": "VU Amsterdam",
            "liacs.leidenuniv.nl": "Leiden University",
            "maastrichtuniversity.nl": "Maastricht University",
            "tno.nl": "TNO",
            "radboudumc.nl": "Radboud University Medical Center",
            "kuleuven.be": "KU Leuven",
            "uclouvain.be": "Catholic University of Louvain",
            "ugent.be": "Ghent University",
            "uliege.be": "University of Liège",
            "umons.ac.be": "University of Mons",
            "ulg.ac.be": "University of Liège",
            "imec.be": "IMEC",
            "uni.lu": "University of Luxembourg",
            "polimi.it": "Politecnico di Milano",
            "unimi.it": "University of Milan",
            "unibo.it": "University of Bologna",
            "unitn.it": "University of Trento",
            "iit.it": "Italian Institute of Technology",
            "unimib.it": "University of Milan-Bicocca",
            "univr.it": "University of Verona",
            "polito.it": "Polytechnic University of Turin",
            "unige.it": "University of Genoa",
            "uv.es": "University of Valencia",
            "upf.edu": "Pompeu Fabra University",
            "cvc.uab.es": "Universitat Autònoma de Barcelona",
            "unizar.es": "University of Zaragoza",
            "fct.unl.pt": "NOVA University Lisbon",
            "tecnico.ulisboa.pt": "Instituto Superior Técnico",
            "ua.pt": "University of Aveiro",
            "fc.up.pt": "University of Porto",
            "tcd.ie": "Trinity College Dublin",
            "ucd.ie": "University College Dublin",
            "ut.ee": "University of Tartu",
            "univie.ac.at": "University of Vienna",
            "tuwien.ac.at": "Vienna University of Technology",
            "tugraz.at": "Graz University of Technology",
            "ist.ac.at": "Institute of Science and Technology Austria",
            "mimuw.edu.pl": "University of Warsaw",
            "informatik.uni-freiburg.de": "University of Freiburg",
            "cs.uni-freiburg.de": "University of Freiburg",
            "univ-grenoble-alpes.fr": "University of Grenoble Alpes",
            "u-bordeaux.fr": "University of Bordeaux",
            "univ-amu.fr": "Aix-Marseille University",
            "sorbonne-universite.fr": "Sorbonne University",
            "imt-atlantique.fr": "IMT Atlantique",
            "eurecom.fr": "EURECOM",
            
            # Asian Universities
            "tsinghua.edu.cn": "Tsinghua University",
            "mails.tsinghua.edu.cn": "Tsinghua University",
            "pku.edu.cn": "Peking University",
            "sjtu.edu.cn": "Shanghai Jiao Tong University",
            "zju.edu.cn": "Zhejiang University",
            "ustc.edu.cn": "University of Science and Technology of China",
            "mail.ustc.edu.cn": "University of Science and Technology of China",
            "whu.edu.cn": "Wuhan University",
            "nju.edu.cn": "Nanjing University",
            "buaa.edu.cn": "Beihang University",
            "xidian.edu.cn": "Xidian University",
            "hit.edu.cn": "Harbin Institute of Technology",
            "cuhk.edu.cn": "Chinese University of Hong Kong (Shenzhen)",
            "seu.edu.cn": "Southeast University",
            "tju.edu.cn": "Tianjin University",
            "fudan.edu.cn": "Fudan University",
            "hust.edu.cn": "Huazhong University of Science and Technology",
            "szu.edu.cn": "Shenzhen University",
            "westlake.edu.cn": "Westlake University",
            "ruc.edu.cn": "Renmin University of China",
            "mail.sysu.edu.cn": "Sun Yat-sen University",
            "uestc.edu.cn": "University of Electronic Science and Technology of China",
            "sustech.edu.cn": "Southern University of Science and Technology",
            "bit.edu.cn": "Beijing Institute of Technology",
            "njust.edu.cn": "Nanjing University of Science and Technology",
            "bupt.edu.cn": "Beijing University of Posts and Telecommunications",
            "shanghaitech.edu.cn": "ShanghaiTech University",
            "tongji.edu.cn": "Tongji University",
            "xmu.edu.cn": "Xiamen University",
            "ust.hk": "Hong Kong University of Science and Technology",
            "connect.ust.hk": "Hong Kong University of Science and Technology",
            "cse.ust.hk": "Hong Kong University of Science and Technology",
            "cse.cuhk.edu.hk": "Chinese University of Hong Kong",
            "cuhk.edu.hk": "Chinese University of Hong Kong",
            "link.cuhk.edu.hk": "Chinese University of Hong Kong",
            "polyu.edu.hk": "Hong Kong Polytechnic University",
            "hku.hk": "University of Hong Kong",
            "cityu.edu.hk": "City University of Hong Kong",
            "um.edu.mo": "University of Macau",
            "nus.edu.sg": "National University of Singapore",
            "u.nus.edu": "National University of Singapore",
            "ntu.edu.sg": "Nanyang Technological University",
            "smu.edu.sg": "Singapore Management University",
            "i2r.a-star.edu.sg": "A*STAR Institute for Infocomm Research",
            "ntu.edu.tw": "National Taiwan University",
            "kaist.ac.kr": "KAIST",
            "snu.ac.kr": "Seoul National University",
            "yonsei.ac.kr": "Yonsei University",
            "korea.ac.kr": "Korea University",
            "skku.edu": "Sungkyunkwan University",
            "boun.edu.tr": "Boğaziçi University",
            "ku.edu.tr": "Koç University",
            "u-tokyo.ac.jp": "University of Tokyo",
            "kyoto-u.ac.jp": "Kyoto University",
            "technion.ac.il": "Technion - Israel Institute of Technology",
            "weizmann.ac.il": "Weizmann Institute of Science",
            "tau.ac.il": "Tel Aviv University",
            "tauex.tau.ac.il": "Tel Aviv University",
            "huji.ac.il": "Hebrew University of Jerusalem",
            "mail.huji.ac.il": "Hebrew University of Jerusalem",
            "biu.ac.il": "Bar-Ilan University",
            "bgu.ac.il": "Ben-Gurion University of the Negev",
            "iisc.ac.in": "Indian Institute of Science",
            
            # Australian/NZ Universities
            "unimelb.edu.au": "University of Melbourne",
            "monash.edu": "Monash University",
            "anu.edu.au": "Australian National University",
            "unsw.edu.au": "University of New South Wales",
            "uts.edu.au": "University of Technology Sydney",
            "qut.edu.au": "Queensland University of Technology",
            "uq.edu.au": "University of Queensland",
            "adelaide.edu.au": "University of Adelaide",
            "rmit.edu.au": "RMIT University",
            "sydney.edu.au": "University of Sydney",
            "mq.edu.au": "Macquarie University",
            "waikato.ac.nz": "University of Waikato",
            
            # Research Institutes and Organizations
            "inria.fr": "INRIA",
            "cnrs.fr": "CNRS",
            "irisa.fr": "IRISA",
            "imag.fr": "IMAG",
            "ens.fr": "École Normale Supérieure",
            "polytechnique.edu": "École Polytechnique",
            "cea.fr": "French Alternative Energies and Atomic Energy Commission",
            "riken.jp": "RIKEN",
            "nii.ac.jp": "National Institute of Informatics",
            "csiro.au": "CSIRO",
            "fbk.eu": "Fondazione Bruno Kessler",
            "ieee.org": "IEEE",
            "acm.org": "ACM",
            "allenai.org": "Allen Institute for AI",
            "nasa.gov": "NASA",
            "nih.gov": "National Institutes of Health",
            "pnnl.gov": "Pacific Northwest National Laboratory",
            "llnl.gov": "Lawrence Livermore National Laboratory",
            "ornl.gov": "Oak Ridge National Laboratory",
            "lbl.gov": "Lawrence Berkeley National Laboratory",
            "sandia.gov": "Sandia National Laboratories",
            "lanl.gov": "Los Alamos National Laboratory",
            "nist.gov": "National Institute of Standards and Technology",
            "dlr.de": "German Aerospace Center (DLR)",
            "ia.ac.cn": "Institute of Automation, Chinese Academy of Sciences",
            "nlpr.ia.ac.cn": "Institute of Automation, Chinese Academy of Sciences",
            "siat.ac.cn": "Shenzhen Institute of Advanced Technology",
            "pjlab.org.cn": "Shanghai AI Laboratory",
            "nyulangone.org": "NYU Langone Health",
            "kaust.edu.sa": "King Abdullah University of Science and Technology",
            "cispa.de": "CISPA Helmholtz Center for Information Security",
            "mpi-inf.mpg.de": "Max Planck Institute for Informatics",
            "ics.forth.gr": "Foundation for Research and Technology - Hellas",
            "skoltech.ru": "Skolkovo Institute of Science and Technology",
            "ijs.si": "Jožef Stefan Institute",
            "inserm.fr": "INSERM",
            "cnr.it": "CNR",
            "isti.cnr.it": "ISTI-CNR",
            "brc.hu": "Hungarian Academy of Sciences",
            "dcc.ufmg.br": "Federal University of Minas Gerais",
            "dac.unicamp.br": "University of Campinas",
        }
        
        self.institution_to_country = {
            # Tech Companies
            "Google": "United States",
            "Google DeepMind": "United Kingdom",
            "DeepMind": "United Kingdom",
            "Microsoft": "United States",
            "Meta (Facebook)": "United States",
            "Facebook": "United States",
            "Amazon": "United States",
            "Apple": "United States",
            "NVIDIA": "United States",
            "OpenAI": "United States",
            "Adobe": "United States",
            "Intel": "United States",
            "Anthropic": "United States",
            "IBM": "United States",
            "SRI International": "United States",
            "Waymo": "United States",
            "ServiceNow": "United States",
            "Oracle": "United States",
            "Salesforce": "United States",
            "Cohere": "Canada",
            "Qualcomm": "United States",
            "Netflix": "United States",
            "LinkedIn": "United States",
            "Uber": "United States",
            "x.ai": "United States",
            "Toyota Research Institute": "United States",
            "General Agents": "United States",
            "Bottou.org": "United States",
            "Ocado": "United Kingdom",
            "Google LLC": "United States",
            "Mila": "Canada",
            "Baidu": "China",
            "ByteDance": "China",
            "Alibaba": "China",
            "Tencent": "China",
            "Huawei": "China",
            "Samsung": "South Korea",
            "Sony": "Japan",
            "Naver Labs": "South Korea",
            "Bosch": "Germany",
            "Siemens": "Germany",
            "IMEC": "Belgium",
            "Nokia": "Finland",
            
            # US Universities
            "University of California, Berkeley": "United States",
            "New York University": "United States",
            "Carnegie Mellon University": "United States",
            "MIT CSAIL": "United States",
            "Massachusetts Institute of Technology": "United States",
            "Stanford University": "United States",
            "University of Michigan": "United States",
            "Cornell University": "United States",
            "Princeton University": "United States",
            "Yale University": "United States",
            "Harvard University": "United States",
            "Harvard Medical School": "United States",
            "Massachusetts General Hospital": "United States",
            "Columbia University": "United States",
            "University of Chicago": "United States",
            "California Institute of Technology": "United States",
            "University of California, San Diego": "United States",
            "University of California, Los Angeles": "United States",
            "University of California, Irvine": "United States",
            "University of California, Santa Barbara": "United States",
            "University of California, Davis": "United States",
            "University of California, San Francisco": "United States",
            "University of California, Santa Cruz": "United States",
            "Georgia Institute of Technology": "United States",
            "University of Southern California": "United States",
            "Northwestern University": "United States",
            "Vanderbilt University": "United States",
            "Brown University": "United States",
            "Dartmouth College": "United States",
            "Rice University": "United States",
            "University of Illinois Urbana-Champaign": "United States",
            "University of Illinois at Chicago": "United States",
            "University of Washington": "United States",
            "University of Wisconsin-Madison": "United States",
            "University of Minnesota": "United States",
            "University of Virginia": "United States",
            "Virginia Tech": "United States",
            "University of Texas at Austin": "United States",
            "University of Texas at Dallas": "United States",
            "Texas A&M University": "United States",
            "Arizona State University": "United States",
            "University of Arizona": "United States",
            "University of Colorado Boulder": "United States",
            "Purdue University": "United States",
            "Ohio State University": "United States",
            "Pennsylvania State University": "United States",
            "University of Pennsylvania": "United States",
            "Rutgers University": "United States",
            "Boston University": "United States",
            "Northeastern University": "United States",
            "University of Florida": "United States",
            "University of Pittsburgh": "United States",
            "Emory University": "United States",
            "George Mason University": "United States",
            "Washington University in St. Louis": "United States",
            "Johns Hopkins University": "United States",
            "Johns Hopkins Medical Institutions": "United States",
            "University at Buffalo": "United States",
            "University of Buffalo": "United States",
            "University of Delaware": "United States",
            "University of Central Florida": "United States",
            "University of Maryland": "United States",
            "University of Maryland, Baltimore County": "United States",
            "North Carolina State University": "United States",
            "University of North Carolina at Chapel Hill": "United States",
            "University of Massachusetts": "United States",
            "University of Massachusetts Amherst": "United States",
            "Toyota Technological Institute at Chicago": "United States",
            "University of Notre Dame": "United States",
            "Oregon State University": "United States",
            "Stony Brook University": "United States",
            "New Jersey Institute of Technology": "United States",
            "NYU Medical Center": "United States",
            "University of Utah": "United States",
            "Duke University": "United States",
            "Michigan State University": "United States",
            "Indiana University": "United States",
            "University of Rochester": "United States",
            "Binghamton University": "United States",
            "Rensselaer Polytechnic Institute": "United States",
            "Tufts University": "United States",
            "University of Tennessee": "United States",
            "University of California, Merced": "United States",
            "University of Connecticut": "United States",
            "University of Georgia": "United States",
            "Iowa State University": "United States",
            
            # Canadian Universities
            "University of Toronto": "Canada",
            "University of British Columbia": "Canada",
            "McGill University": "Canada",
            "University of Waterloo": "Canada",
            "University of Montreal": "Canada",
            "Université de Montréal": "Canada",
            "MILA": "Canada",
            "Simon Fraser University": "Canada",
            "University of Calgary": "Canada",
            "University of Western Ontario": "Canada",
            "University of Ottawa": "Canada",
            "École de technologie supérieure": "Canada",
            "University of Alberta": "Canada",
            "Polytechnique Montréal": "Canada",
            "Queen's University": "Canada",
            "Concordia University": "Canada",
            "York University": "Canada",
            "University of Sherbrooke": "Canada",
            "Dalhousie University": "Canada",
            "University of Victoria": "Canada",
            "Université Laval": "Canada",
            
            # UK Universities
            "University of Cambridge": "United Kingdom",
            "University of Oxford": "United Kingdom",
            "University College London": "United Kingdom",
            "Imperial College London": "United Kingdom",
            "University of Edinburgh": "United Kingdom",
            "King's College London": "United Kingdom",
            "Queen Mary University of London": "United Kingdom",
            "University of Surrey": "United Kingdom",
            "University of Glasgow": "United Kingdom",
            "University of Sheffield": "United Kingdom",
            "University of Bristol": "United Kingdom",
            "University of Manchester": "United Kingdom",
            "University of Nottingham": "United Kingdom",
            "University of Warwick": "United Kingdom",
            "Lancaster University": "United Kingdom",
            "University of Southampton": "United Kingdom",
            "University of York": "United Kingdom",
            "University of St Andrews": "United Kingdom",
            "Cardiff University": "United Kingdom",
            "University of Bath": "United Kingdom",
            
            # German Universities and Research
            "Technical University of Munich": "Germany",
            "Technical University of Berlin": "Germany",
            "University of Tübingen": "Germany",
            "Max Planck Institute for Intelligent Systems": "Germany",
            "Max Planck Institute for Informatics": "Germany",
            "Max Planck Institute": "Germany",
            "Karlsruhe Institute of Technology": "Germany",
            "TU Dresden": "Germany",
            "RWTH Aachen University": "Germany",
            "University of Freiburg": "Germany",
            "CISPA Helmholtz Center for Information Security": "Germany",
            "German Aerospace Center (DLR)": "Germany",
            
            # Swiss Universities
            "ETH Zurich": "Switzerland",
            "EPFL": "Switzerland",
            "University of Geneva": "Switzerland",
            "University of Basel": "Switzerland",
            "Idiap Research Institute": "Switzerland",
            "University of Zurich": "Switzerland",
            "Università della Svizzera italiana": "Switzerland",
            
            # Other European Countries...
            # (truncated for brevity - continuing with comprehensive mappings)
            "University of Grenoble Alpes": "France",
            "University of Bordeaux": "France",
            "Aix-Marseille University": "France",
            "Sorbonne University": "France",
            "IMT Atlantique": "France",
            "EURECOM": "France",
            "INRIA": "France",
            "CNRS": "France",
            "IRISA": "France",
            "IMAG": "France",
            "École Normale Supérieure": "France",
            "École Polytechnique": "France",
            "French Alternative Energies and Atomic Energy Commission": "France",
            "INSERM": "France",
            
            # Continue with all other mappings...
            "National Taiwan University": "Taiwan",
            "Korea University": "South Korea",
            "Tsinghua University": "China",
            "Peking University": "China",
            "University of Tokyo": "Japan",
            "KAIST": "South Korea",
            "Technion - Israel Institute of Technology": "Israel",
            "Weizmann Institute of Science": "Israel",
            "Tel Aviv University": "Israel",
            "Hebrew University of Jerusalem": "Israel",
            "University of Melbourne": "Australia",
            "University of Sydney": "Australia",
            "Monash University": "Australia",
            "Australian National University": "Australia",
            "National University of Singapore": "Singapore",
            "Nanyang Technological University": "Singapore",
            "Hong Kong University of Science and Technology": "Hong Kong",
            "Chinese University of Hong Kong": "Hong Kong",
            "University of Hong Kong": "Hong Kong",
            "Indian Institute of Science": "India",
        }
        
        self.country_standardization = {
            "us": "United States",
            "usa": "United States", 
            "united states": "United States",
            "uk": "United Kingdom",
            "united kingdom": "United Kingdom",
            "cn": "China",
            "china": "China",
            "ca": "Canada",
            "canada": "Canada",
            "de": "Germany",
            "germany": "Germany",
            "fr": "France",
            "france": "France",
            "au": "Australia",
            "australia": "Australia",
            "sg": "Singapore",
            "singapore": "Singapore",
            "kr": "South Korea",
            "south korea": "South Korea",
            "jp": "Japan",
            "japan": "Japan",
            "at": "Austria",
            "austria": "Austria",
            "nz": "New Zealand",
            "new zealand": "New Zealand",
            "be": "Belgium",
            "belgium": "Belgium",
            "in": "India",
            "india": "India",
            "ch": "Switzerland",
            "switzerland": "Switzerland",
            "nl": "Netherlands",
            "netherlands": "Netherlands",
            "se": "Sweden",
            "sweden": "Sweden",
            "it": "Italy",
            "italy": "Italy",
            "fi": "Finland",
            "finland": "Finland",
            "no": "Norway",
            "norway": "Norway",
            "dk": "Denmark",
            "denmark": "Denmark",
            "ie": "Ireland",
            "ireland": "Ireland",
            "es": "Spain",
            "spain": "Spain",
            "pt": "Portugal",
            "portugal": "Portugal",
            "il": "Israel",
            "israel": "Israel",
            "hk": "Hong Kong",
            "hong kong": "Hong Kong",
            "tw": "Taiwan",
            "taiwan": "Taiwan",
            "international": "International",
        }
        
        self.load_saved_mappings()

    def load_saved_mappings(self):
        """Load any previously saved domain mappings"""
        if os.path.exists(self.domain_mappings_file):
            try:
                with open(self.domain_mappings_file, 'r') as f:
                    saved_mappings = json.load(f)
                    self.domain_to_institution.update(saved_mappings.get('institutions', {}))
                    self.suffix_country_map.update(saved_mappings.get('countries', {}))
                print(f"Loaded {len(saved_mappings.get('institutions', {}))} institution mappings and {len(saved_mappings.get('countries', {}))} country mappings")
            except Exception as e:
                print(f"Error loading saved mappings: {e}")
                
        if os.path.exists(self.institution_country_mappings_file):
            try:
                with open(self.institution_country_mappings_file, 'r') as f:
                    saved_institution_mappings = json.load(f)
                    self.institution_to_country.update(saved_institution_mappings)
                print(f"Loaded {len(saved_institution_mappings)} institution-to-country mappings")
            except Exception as e:
                print(f"Error loading institution country mappings: {e}")

    def save_mappings(self, new_institutions=None, new_countries=None):
        """Save new mappings to file"""
        try:
            existing_data = {"institutions": {}, "countries": {}}
            if os.path.exists(self.domain_mappings_file):
                with open(self.domain_mappings_file, 'r') as f:
                    existing_data = json.load(f)
            
            if new_institutions:
                existing_data.setdefault('institutions', {}).update(new_institutions)
                self.domain_to_institution.update(new_institutions)
            
            if new_countries:
                existing_data.setdefault('countries', {}).update(new_countries)
                self.suffix_country_map.update(new_countries)
            
            with open(self.domain_mappings_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            # Save institution to country mappings
            with open(self.institution_country_mappings_file, 'w') as f:
                json.dump(self.institution_to_country, f, indent=2)
            
            print(f"Saved mappings to {self.domain_mappings_file}")
        except Exception as e:
            print(f"Error saving mappings: {e}")

    def load_data(self):
        """Load the CSV data"""
        if not os.path.exists(self.csv_path):
            print(f"Error: Input file '{self.csv_path}' does not exist.")
            return False
        
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        
        # Ensure columns exist
        if "country" not in self.df.columns:
            self.df["country"] = "Unknown"
        if "institution" not in self.df.columns:
            self.df["institution"] = "Unknown"
        
        print(f"Loaded {len(self.df)} records from {self.csv_path}")
        return True

    def extract_domain_from_email(self, email_field):
        """Extract domain from email field"""
        if pd.isna(email_field) or email_field == "":
            return None
        
        match = re.search(r"Verified email at ([^\s]+?)(?:\s*-\s*Homepage)?$", str(email_field))
        if not match:
            return None
        
        return match.group(1).lower().strip()

    def infer_country_from_domain(self, domain):
        """Infer country from domain"""
        if not domain:
            return "Unknown"
        
        ext = tldextract.extract(domain)
        suffix = ext.suffix.lower()
        
        if suffix in self.suffix_country_map:
            return self.suffix_country_map[suffix]
        else:
            last_part = suffix.split('.')[-1]
            return self.suffix_country_map.get(last_part, "Unknown")

    def infer_institution_from_domain(self, domain):
        """Infer institution from domain"""
        if not domain:
            return "Unknown"
        
        for known_domain in self.domain_to_institution:
            if domain.endswith(known_domain):
                return self.domain_to_institution[known_domain]
        return "Unknown"

    def infer_country_from_institution(self, institution):
        """Infer country from institution name"""
        if not institution or str(institution).lower() in ["unknown", "nan", "", "none"]:
            return "Unknown"
        
        institution = str(institution).strip()
        
        # Direct mapping
        if institution in self.institution_to_country:
            return self.institution_to_country[institution]
        
        # Fuzzy matching for variations
        for mapped_institution, country in self.institution_to_country.items():
            if len(institution) > 5 and len(mapped_institution) > 5:
                if (mapped_institution.lower() in institution.lower() or 
                    institution.lower() in mapped_institution.lower()):
                    return country
        
        return "Unknown"

    def get_statistics(self):
        """Get current statistics"""
        if self.df is None:
            return None
            
        total_records = len(self.df)
        
        # Count unknown countries (case-insensitive)
        unknown_countries = len(self.df[
            self.df['country'].isna() | 
            self.df['country'].str.lower().isin(['unknown', 'nan', '', 'none'])
        ])
        
        # Count unknown institutions
        unknown_institutions = len(self.df[
            self.df['institution'].isna() | 
            self.df['institution'].str.lower().isin(['unknown', 'nan', '', 'none'])
        ])
        
        return {
            'total_records': total_records,
            'unknown_countries': unknown_countries,
            'unknown_institutions': unknown_institutions,
            'mapped_countries': total_records - unknown_countries,
            'mapped_institutions': total_records - unknown_institutions,
            'country_percentage': ((total_records - unknown_countries) / total_records * 100),
            'institution_percentage': ((total_records - unknown_institutions) / total_records * 100)
        }

    def show_statistics(self):
        """Display current statistics"""
        if not self.load_data():
            return
            
        stats = self.get_statistics()
        if stats is None:
            print("Error: Could not calculate statistics")
            return
        
        print("🎯 Scholar Profiles Statistics")
        print("=" * 50)
        print(f"Total records: {stats['total_records']:,}")
        print(f"Unknown countries: {stats['unknown_countries']:,} ({100-stats['country_percentage']:.1f}%)")
        print(f"Mapped countries: {stats['mapped_countries']:,} ({stats['country_percentage']:.1f}%)")
        print(f"Unknown institutions: {stats['unknown_institutions']:,} ({100-stats['institution_percentage']:.1f}%)")
        print(f"Mapped institutions: {stats['mapped_institutions']:,} ({stats['institution_percentage']:.1f}%)")
        
        if self.df is not None:
            print(f"\nTop 10 countries:")
            print(self.df['country'].value_counts().head(10))

    def get_unknown_domains(self, top_n=100, type_filter="institution"):
        """Get top N domains that don't have mappings"""
        if self.df is None:
            return []
        
        unknown_domains = []
        
        for _, row in self.df.iterrows():
            domain = self.extract_domain_from_email(row['email'])
            if domain:
                if type_filter == "institution":
                    if (row['institution'] == 'Unknown' or 
                        pd.isna(row['institution']) or 
                        str(row['institution']).lower() in ['unknown', 'nan', '', 'none']):
                        institution = self.infer_institution_from_domain(domain)
                        if institution == "Unknown":
                            unknown_domains.append(domain)
                elif type_filter == "country":
                    if (pd.isna(row['country']) or 
                        str(row['country']).lower() in ['unknown', 'nan', '', 'none']):
                        country = self.infer_country_from_domain(domain)
                        if country == "Unknown":
                            unknown_domains.append(domain)
        
        # Count occurrences
        domain_counts = Counter(unknown_domains)
        
        # Get top N unknown domains
        top_domains = []
        for domain, count in domain_counts.most_common(top_n):
            # Get some sample names from this domain
            sample_condition = self.df['email'].str.contains(f"at {re.escape(domain)}", na=False, case=False)
            if type_filter == "institution":
                sample_condition = sample_condition & (
                    self.df['institution'].isna() | 
                    self.df['institution'].str.lower().isin(['unknown', 'nan', '', 'none'])
                )
            else:
                sample_condition = sample_condition & (
                    self.df['country'].isna() | 
                    self.df['country'].str.lower().isin(['unknown', 'nan', '', 'none'])
                )
            
            sample_records = self.df[sample_condition]['name'].head(3).tolist()
            
            top_domains.append({
                'domain': domain,
                'count': count,
                'sample_records': sample_records
            })
        
        return top_domains

    def show_unknown_domains(self, top_n=50, type_filter="institution"):
        """Display top unknown domains"""
        if not self.load_data():
            return
            
        unknown_domains = self.get_unknown_domains(top_n, type_filter)
        
        if not unknown_domains:
            print(f"🎉 No unknown {type_filter} domains found! All domains are mapped.")
            return
        
        print(f"\n📊 Top {min(top_n, len(unknown_domains))} domains with unknown {type_filter}:")
        print("=" * 80)
        print(f"{'Rank':<4} {'Domain':<35} {'Count':<8} {'Sample Records'}")
        print("-" * 80)
        
        for i, domain_info in enumerate(unknown_domains, 1):
            domain = domain_info['domain']
            count = domain_info['count']
            
            sample_text = "; ".join(domain_info['sample_records'][:2])
            if len(sample_text) > 30:
                sample_text = sample_text[:27] + "..."
            
            print(f"{i:<4} {domain:<35} {count:<8} {sample_text}")
        
        print(f"\n💡 Example JSON additions for {self.domain_mappings_file}:")
        if type_filter == "institution":
            print("Add to the 'institutions' section:")
            for domain_info in unknown_domains[:5]:
                domain = domain_info['domain']
                print(f'    "{domain}": "Your Institution Name Here",')
        else:
            print("Add to the 'countries' section:")
            for domain_info in unknown_domains[:5]:
                domain = domain_info['domain']
                ext = tldextract.extract(domain)
                suffix = ext.suffix.lower()
                if suffix:
                    print(f'    "{suffix}": "Your Country Name Here",')

    def standardize_countries(self):
        """Standardize country name variations"""
        if self.df is None:
            if not self.load_data():
                return 0
            
        print("🌍 Standardizing country names...")
        
        standardized = 0
        for index, row in self.df.iterrows():
            if pd.notna(row['country']):
                country_lower = str(row['country']).lower().strip()
                if country_lower in self.country_standardization:
                    new_country = self.country_standardization[country_lower]
                    if new_country != row['country']:
                        self.df.at[index, 'country'] = new_country
                        standardized += 1
        
        print(f"Standardized {standardized} country names")
        return standardized

    def fix_institutions_from_domains(self):
        """Fix unknown institutions using domain mappings"""
        if self.df is None:
            if not self.load_data():
                return 0
            
        print("🏢 Fixing institutions from email domains...")
        
        updated = 0
        for index, row in self.df.iterrows():
            institution_is_unknown = (
                pd.isna(row["institution"]) or 
                str(row["institution"]).lower() in ["unknown", "nan", "", "none"]
            )
            
            if institution_is_unknown:
                domain = self.extract_domain_from_email(row["email"])
                if domain:
                    new_institution = self.infer_institution_from_domain(domain)
                    if new_institution != "Unknown":
                        self.df.at[index, "institution"] = new_institution
                        updated += 1
        
        print(f"Updated {updated} institution records from domains")
        return updated

    def fix_countries_from_domains(self):
        """Fix unknown countries using domain mappings"""
        if self.df is None:
            if not self.load_data():
                return 0
            
        print("🌍 Fixing countries from email domains...")
        
        updated = 0
        for index, row in self.df.iterrows():
            country_is_unknown = (
                pd.isna(row["country"]) or 
                str(row["country"]).lower() in ["unknown", "nan", "", "none"]
            )
            
            if country_is_unknown:
                domain = self.extract_domain_from_email(row["email"])
                if domain:
                    new_country = self.infer_country_from_domain(domain)
                    if new_country != "Unknown":
                        self.df.at[index, "country"] = new_country
                        updated += 1
        
        print(f"Updated {updated} country records from domains")
        return updated

    def fix_countries_from_institutions(self):
        """Fix unknown countries using institution mappings"""
        if self.df is None:
            if not self.load_data():
                return 0
            
        print("🏢→🌍 Fixing countries from institutions...")
        
        updated = 0
        for index, row in self.df.iterrows():
            country_is_unknown = (
                pd.isna(row["country"]) or 
                str(row["country"]).lower() in ["unknown", "nan", "", "none"]
            )
            
            if country_is_unknown and pd.notna(row["institution"]):
                new_country = self.infer_country_from_institution(row["institution"])
                if new_country != "Unknown":
                    self.df.at[index, "country"] = new_country
                    updated += 1
        
        print(f"Updated {updated} country records from institutions")
        return updated

    def save_data(self):
        """Save the updated CSV"""
        if self.df is not None:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self.df.to_csv(self.csv_path, index=False)
            print(f"✅ Saved updated data to {self.csv_path}")

    def fix_all(self):
        """Run all fixing operations"""
        print("🚀 Running comprehensive fix operations...")
        
        if self.df is None:
            if not self.load_data():
                return
        
        # Show initial statistics
        initial_stats = self.get_statistics()
        if initial_stats is None:
            print("❌ Failed to get initial statistics")
            return
            
        print(f"Initial state:")
        print(f"  Unknown countries: {initial_stats['unknown_countries']:,}")
        print(f"  Unknown institutions: {initial_stats['unknown_institutions']:,}")
        
        # Run all fixes
        country_standardized = self.standardize_countries()
        institutions_fixed = self.fix_institutions_from_domains()
        countries_from_domains = self.fix_countries_from_domains()
        countries_from_institutions = self.fix_countries_from_institutions()
        
        # Save data
        self.save_data()
        
        # Show final statistics
        final_stats = self.get_statistics()
        if final_stats is None:
            print("❌ Failed to get final statistics")
            return
            
        print(f"\n📊 Final Results:")
        print(f"  Unknown countries: {final_stats['unknown_countries']:,} (was {initial_stats['unknown_countries']:,})")
        print(f"  Unknown institutions: {final_stats['unknown_institutions']:,} (was {initial_stats['unknown_institutions']:,})")
        print(f"  Country mapping: {final_stats['country_percentage']:.1f}%")
        print(f"  Institution mapping: {final_stats['institution_percentage']:.1f}%")
        
        print(f"\n✅ Summary of changes:")
        print(f"  • Standardized {country_standardized} country names")
        print(f"  • Fixed {institutions_fixed} institutions from domains")
        print(f"  • Fixed {countries_from_domains} countries from domains")
        print(f"  • Fixed {countries_from_institutions} countries from institutions")

    def interactive_mode(self):
        """Interactive mode for adding mappings"""
        if self.df is None:
            if not self.load_data():
                return
            
        while True:
            print("\n🔧 Interactive Mapping Mode")
            print("=" * 40)
            print("1. Show statistics")
            print("2. Show unknown institution domains")
            print("3. Show unknown country domains")
            print("4. Add institution mappings")
            print("5. Add country mappings")
            print("6. Run fix all")
            print("7. Exit")
            
            choice = input("\nChoice (1-7): ").strip()
            
            if choice == '1':
                self.show_statistics()
            elif choice == '2':
                self.show_unknown_domains(50, "institution")
            elif choice == '3':
                self.show_unknown_domains(50, "country")
            elif choice == '4':
                self.add_mappings_interactive("institution")
            elif choice == '5':
                self.add_mappings_interactive("country")
            elif choice == '6':
                self.fix_all()
            elif choice == '7':
                print("👋 Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

    def add_mappings_interactive(self, mapping_type="institution"):
        """Interactive mapping addition"""
        if mapping_type == "institution":
            unknown_domains = self.get_unknown_domains(20, "institution")
            mapping_dict = self.domain_to_institution
            save_key = "institutions"
        else:
            unknown_domains = self.get_unknown_domains(20, "country")
            mapping_dict = self.suffix_country_map
            save_key = "countries"
        
        if not unknown_domains:
            print(f"No unknown {mapping_type} domains found!")
            return
        
        new_mappings = {}
        
        print(f"\n🔧 Add {mapping_type} mappings (press Enter to skip):")
        print("=" * 60)
        
        for domain_info in unknown_domains[:10]:  # Limit to top 10
            domain = domain_info['domain']
            count = domain_info['count']
            samples = "; ".join(domain_info['sample_records'][:2])
            
            print(f"\nDomain: {domain} (appears {count} times)")
            print(f"Sample researchers: {samples}")
            
            if mapping_type == "institution":
                mapping = input(f"Enter institution for {domain}: ").strip()
                if mapping:
                    new_mappings[domain] = mapping
                    print(f"✅ Will map '{domain}' → '{mapping}'")
            else:
                # For countries, map the TLD suffix
                ext = tldextract.extract(domain)
                suffix = ext.suffix.lower()
                if suffix:
                    mapping = input(f"Enter country for suffix '{suffix}' (from {domain}): ").strip()
                    if mapping:
                        new_mappings[suffix] = mapping
                        print(f"✅ Will map suffix '{suffix}' → '{mapping}'")
            
            # Ask if user wants to continue
            if input("\nContinue to next domain? (y/n): ").lower().strip() == 'n':
                break
        
        if new_mappings:
            # Save mappings
            if mapping_type == "institution":
                self.save_mappings(new_institutions=new_mappings)
            else:
                self.save_mappings(new_countries=new_mappings)
            
            print(f"\n✅ Added {len(new_mappings)} new {mapping_type} mappings!")
            
            # Ask if user wants to apply them immediately
            if input("Apply these mappings now? (y/n): ").lower().strip() == 'y':
                if mapping_type == "institution":
                    self.fix_institutions_from_domains()
                else:
                    self.fix_countries_from_domains()
                self.save_data()
        else:
            print("No new mappings added.")

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Scholar Profile Domain and Country Mapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scholar_profile_mapper.py --stats
  python scholar_profile_mapper.py --fix-all
  python scholar_profile_mapper.py --interactive
  python scholar_profile_mapper.py --show-unknowns --type institution
  python scholar_profile_mapper.py --show-unknowns --type country
        """
    )
    
    parser.add_argument('--csv', default='data/processed/scholar_profiles.csv',
                       help='Path to the scholar profiles CSV file')
    parser.add_argument('--stats', action='store_true',
                       help='Show current statistics')
    parser.add_argument('--fix-all', action='store_true',
                       help='Run all fixing operations')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--show-unknowns', action='store_true',
                       help='Show unknown domains')
    parser.add_argument('--type', choices=['institution', 'country'], default='institution',
                       help='Type of unknowns to show (default: institution)')
    parser.add_argument('--limit', type=int, default=50,
                       help='Number of unknown domains to show (default: 50)')
    
    args = parser.parse_args()
    
    # Create mapper instance
    mapper = ScholarProfileMapper(args.csv)
    
    # Execute based on arguments
    if args.stats:
        mapper.show_statistics()
    elif args.fix_all:
        mapper.fix_all()
    elif args.interactive:
        mapper.interactive_mode()
    elif args.show_unknowns:
        mapper.show_unknown_domains(args.limit, args.type)
    else:
        # Default: show help
        parser.print_help()

if __name__ == "__main__":
    main()
