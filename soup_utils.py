import requests
from bs4 import BeautifulSoup
import pickle
from io import BytesIO
import os
import hashlib


def get_page_soup(url):
    # Figure out if we have a cached version (filename = md5 hash of url)
    hash_hex = hashlib.md5(url.encode('utf-8')).hexdigest()
    cached_page_path = "./cached_pages/" + str(hash_hex) + ".html"

    if os.path.exists(cached_page_path):
        print("Using cached page!")
        with open(cached_page_path, "rb") as f:
            content = BytesIO(f.read()).read()
    else:
        print("Downloading page...")
        page = requests.get(url)
        content = page.content
        with open(cached_page_path, "wb") as f:
            f.write(content)

    return BeautifulSoup(content, "html.parser")
