import os
import time
from google_images_download import google_images_download

# Set the base directory where your empty folders are located
base_directory = '~/Downloads/proj_garb_classify/img/'
base_directory = os.path.expanduser(base_directory)

def download_images(label, output_path, num_images=100, retries=3):
    response = google_images_download.googleimagesdownload()
    
    for attempt in range(retries):
        try:
            # Try different search terms
            if attempt == 0:
                keywords = label
            elif attempt == 1:
                keywords = f"{label} object"
            else:
                keywords = f"{label} item photo"

            arguments = {
                "keywords": keywords,
                "limit": num_images,
                "print_urls": False,
                "format": "jpg",
                "size": "medium",  # Try medium size instead of icon
                "type": "photo",
                "output_directory": output_path,
                "no_directory": True,
                "prefix": label + "_",
                "safe_search": True,
                "delay": 2,  # Add a delay between downloads
            }
            
            paths = response.download(arguments)
            downloaded = len(paths[0].get(keywords, []))
            print(f"Downloaded {downloaded} images for {label} (Attempt {attempt + 1})")
            
            if downloaded > 0:
                return
            
        except Exception as e:
            print(f"Error downloading {label}: {str(e)}")
        
        time.sleep(5)  # Wait before retrying
    
    print(f"Failed to download images for {label} after {retries} attempts")

# Iterate through all subdirectories in the base directory
for root, dirs, files in os.walk(base_directory):
    for dir in dirs:
        category_path = os.path.join(root, dir)
        if os.path.isdir(category_path):
            for subfolder in os.listdir(category_path):
                subfolder_path = os.path.join(category_path, subfolder)
                if os.path.isdir(subfolder_path):
                    label = subfolder
                    print(f"Attempting to download images for: {label}")
                    download_images(label, subfolder_path)

print("Image download complete!")
