import os
import re
import sys
import requests
from urllib.parse import urlparse

PDF_REGEX = re.compile(
    r"https://media\.harel-group\.co\.il/media/[^\"<> \']+\.pdf",
    re.IGNORECASE
)

IS_DOWNLOAD = False

def find_pdfs_in_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return PDF_REGEX.findall(f.read())
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return []

def download_pdf(url, output_dir, base_path_origin):
    filename = os.path.basename(urlparse(url).path)
    dir_structure = os.path.join(output_dir, base_path_origin)
    output_path = os.path.join(dir_structure, filename)

    if os.path.exists(output_path):
        return  # already downloaded

    # create detailed dir structure
    os.makedirs(dir_structure, exist_ok=True)

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def scan_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seen = set()

    for root, _, files in os.walk(input_dir):
        for name in files:
            file_path = os.path.join(root, name)
            pdfs = find_pdfs_in_file(file_path)

            for pdf in pdfs:
                if pdf not in seen:
                    #import ipdb;ipdb.set_trace()
                    base_path_origin = "www.harel-group.co.il"+file_path.split("www.harel-group.co.il")[1]
                    print(base_path_origin + "," + pdf)
                    seen.add(pdf)
                    #print(pdf)
                    if IS_DOWNLOAD:
                        download_pdf(pdf, output_dir, base_path_origin)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scan_and_download_pdfs.py <input_dir> <output_dir>")
        sys.exit(1)

    scan_directory(sys.argv[1], sys.argv[2])
