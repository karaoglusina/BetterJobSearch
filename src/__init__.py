"""SinaJobs - job search automation tool that scrapes LinkedIn jobs
using Apify API and stores them in MongoDB for analysis.
"""

__version__ = "0.1.0"
__author__ = "Sina Karaoglu"

# Main modules
from . import config
from . import scraper
from . import database

__all__ = ['config', 'scraper', 'database']