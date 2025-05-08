# nltk_download.py
import nltk
import os

# Download directory path
download_dir = os.path.join(os.path.dirname(__file__), "nltk_data")

# Make sure the directory exists
os.makedirs(download_dir, exist_ok=True)

# Download the necessary NLTK data
nltk.download("punkt", download_dir=download_dir)
nltk.download("punkt_tab", download_dir=download_dir)
nltk.download("wordnet", download_dir=download_dir)

print(f"NLTK data downloaded to: {download_dir}")