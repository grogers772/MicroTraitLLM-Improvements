import csv
import requests
import tarfile
import io
import os
import shutil
from bs4 import BeautifulSoup
from datetime import datetime as dt
import re
# This logic works for a list of PMCIDs. We also now can work with a heavy download. Also convert vecablation_topn.py
# to use this method, download all 20 articles to be used for the single question, then only use a certain number of articles at a time?

def parse_ftp_directory(url):
    """Parse the FTP directory listing and extract tar.gz and csv files"""
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links to tar.gz and csv files
    tar_files = []
    csv_files = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            if href.endswith('.tar.gz'):
                tar_files.append(href)
            elif href.endswith('.filelist.csv'):
                csv_files.append(href)
    
    return tar_files, csv_files

def get_csv_filename(tar_filename):
    """Convert tar.gz filename to corresponding csv filename"""
    # Replace .tar.gz with .filelist.csv
    return tar_filename.replace('.tar.gz', '.filelist.csv')

def extract_date_from_filename(filename):
    """Extract date from filename in format YYYY-MM-DD"""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return dt.strptime(match.group(1), '%Y-%m-%d')
    return None

def check_files_exist(filenames, output_dir='downloads'):
    """Check if all files in the list already exist"""
    if not os.path.exists(output_dir):
        return False
    
    for filename in filenames:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            return False
    return True

def download_file(url, filename, output_dir='downloads'):
    """Download a file with progress indication"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Check if file already exists
    if os.path.exists(filepath):
        print(f"⏭ Skipping {filename} (already exists)")
        return filepath
    
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                progress = (downloaded / total_size) * 100
                print(f"\rProgress: {progress:.1f}%", end='')
    
    print(f"\n✓ Downloaded: {filename}")
    return filepath

def main():
    base_urls = ["https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/","https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/","https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/other/xml/"]
    output_dir = 'FTP_Downloads'
    
    print("Fetching directory listing...")
    for base_url in base_urls:
        tar_files, csv_files = parse_ftp_directory(base_url)
        
        print(f"Found {len(tar_files)} tar.gz files")
        print(f"Found {len(csv_files)} csv files\n")
        
        # Separate baseline and incremental files
        base_files = [f for f in tar_files if 'baseline' in f]
        incr_files = [f for f in tar_files if 'incr' in f and 'baseline' not in f]
        
        print(f"Total baseline files found: {len(base_files)}")
        print(f"Incremental files: {len(incr_files)}\n")
        
        # Group baseline files by date
        baseline_by_date = {}
        for filename in base_files:
            date = extract_date_from_filename(filename)
            if date:
                date_str = date.strftime('%Y-%m-%d')
                if date_str not in baseline_by_date:
                    baseline_by_date[date_str] = []
                baseline_by_date[date_str].append(filename)
        
        # Check baseline dates and select most recent
        base_files_to_download = []
        baseline_csv_files = []
        if baseline_by_date:
            print("Baseline dates found:")
            for date_str in sorted(baseline_by_date.keys()):
                print(f"  {date_str}: {len(baseline_by_date[date_str])} files")
            
            # Get the most recent date
            most_recent_baseline_date = max(baseline_by_date.keys())
            base_files_to_download = baseline_by_date[most_recent_baseline_date]
            
            # Get corresponding CSV files
            for tar_file in base_files_to_download:
                csv_file = get_csv_filename(tar_file)
                if csv_file in csv_files:
                    baseline_csv_files.append(csv_file)
            
            print(f"\nMost recent baseline date: {most_recent_baseline_date}")
            print(f"Files to download: {len(base_files_to_download)} tar.gz + {len(baseline_csv_files)} csv\n")
        
        # Find and prepare the most recent incremental file
        incr_file_to_download = None
        incr_csv_file = None
        if incr_files:
            # Extract dates and find the most recent
            incr_with_dates = []
            for filename in incr_files:
                date = extract_date_from_filename(filename)
                if date:
                    incr_with_dates.append((filename, date))
            
            if incr_with_dates:
                # Sort by date and get the most recent
                incr_with_dates.sort(key=lambda x: x[1], reverse=True)
                incr_file_to_download, most_recent_date = incr_with_dates[0]
                
                csv_file = get_csv_filename(incr_file_to_download)
                if csv_file in csv_files:
                    incr_csv_file = csv_file
                
                print(f"Most recent incremental file: {incr_file_to_download}")
                print(f"Date: {most_recent_date.strftime('%Y-%m-%d')}\n")
        
        # Collect all files that need to be downloaded
        all_files_needed = base_files_to_download + baseline_csv_files
        if incr_file_to_download:
            all_files_needed.append(incr_file_to_download)
        if incr_csv_file:
            all_files_needed.append(incr_csv_file)
        
        # Check if all files already exist
        if check_files_exist(all_files_needed, output_dir):
            print("=" * 60)
            print("ALL FILES ALREADY EXIST")
            print("=" * 60)
            print(f"All {len(all_files_needed)} required files are already in '{output_dir}/'")
            print("Skipping download process.\n")
            return
        
        # Download baseline files
        if base_files_to_download:
            print("=" * 60)
            print("DOWNLOADING BASELINE FILES")
            print("=" * 60)
            for filename in base_files_to_download:
                file_url = base_url + filename
                download_file(file_url, filename, output_dir)
                
                # Download corresponding CSV file
                csv_filename = get_csv_filename(filename)
                if csv_filename in csv_files:
                    csv_url = base_url + csv_filename
                    download_file(csv_url, csv_filename, output_dir)
                else:
                    print(f"  ⚠ Warning: CSV file not found for {filename}")
        else:
            print("No baseline files with valid dates found.")
        
        # Download the most recent incremental file
        if incr_file_to_download:
            print("\n" + "=" * 60)
            print("DOWNLOADING MOST RECENT INCREMENTAL FILE")
            print("=" * 60)
            
            file_url = base_url + incr_file_to_download
            download_file(file_url, incr_file_to_download, output_dir)
            
            # Download corresponding CSV file
            if incr_csv_file:
                csv_url = base_url + incr_csv_file
                download_file(csv_url, incr_csv_file, output_dir)
            else:
                print(f"  ⚠ Warning: CSV file not found for {incr_file_to_download}")
        else:
            print("No incremental files with valid dates found.")
        
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE")
        print("=" * 60)

if __name__ == "__main__":
    main()