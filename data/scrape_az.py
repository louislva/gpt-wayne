import requests
from bs4 import BeautifulSoup
import pickle
from soup_utils import get_page_soup, delete_cache

DOMAIN = "https://www.azlyrics.com"
URL = DOMAIN + "/l/lilwayne.html"

soup = get_page_soup(URL)
song_links = [a for a in soup.select(".listalbum-item>a")]


# Scrape every song lyrics for each link in song_links
for song_link in song_links.reverse():
    print(DOMAIN + song_link.get("href")[2:])
    song_soup = get_page_soup(DOMAIN + song_link.get("href")[2:])

    try:
        # Extract all text from the html page
        lyrics = song_soup.find(
            "div", {"class": "col-xs-12 col-lg-8 text-center"}
        ).get_text()
    except Exception as e:
        delete_cache(DOMAIN + song_link.get("href")[2:])
        raise e
    print(lyrics)


# song_lyrics = song_soup.select(".col-xs-12.col-lg-8.text-center>br ~ div")[0]
