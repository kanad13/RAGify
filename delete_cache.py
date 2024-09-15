# Script to delete caching files

import os

def delete_files(file_paths):
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"File '{file_path}' has been deleted successfully.")
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except PermissionError:
            print(f"Permission denied: unable to delete '{file_path}'.")
        except Exception as e:
            print(f"An error occurred with '{file_path}': {e}")


# List of files to be deleted
files_to_delete = [
    './semantic_cache.json',
    './faiss_index.pkl',
]

delete_files(files_to_delete)
