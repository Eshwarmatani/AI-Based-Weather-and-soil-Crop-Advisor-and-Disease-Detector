import time
import kagglehub

def download_with_retry(dataset_name, max_retries=10, delay=5):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n[Attempt {attempt}/{max_retries}] Downloading {dataset_name}...")
            path = kagglehub.dataset_download(dataset_name)
            print(f"Successfully downloaded {dataset_name}!")
            return path
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                sleep_time = delay * attempt
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"All {max_retries} attempts failed for {dataset_name}.")
                raise

if __name__ == "__main__":
    crop_path = download_with_retry("atharvaingle/crop-recommendation-dataset")
    print("Path to crop recommendation dataset files:", crop_path)

    plant_path = download_with_retry("emmarex/plantdisease")
    print("Path to plant disease dataset files:", plant_path)

    rice_path = download_with_retry("anshulm257/rice-disease-dataset")
    print("Path to rice disease dataset files:", rice_path)
