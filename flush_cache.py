# Script Purpose - Cache Management and File Deletion
# This script allows the user to choose between clearing Streamlit's cache,
# deleting specific caching files, or doing both.
# To execute, run the script manually with `python3 flush_caches.py`

import streamlit as st
import os

def flush_cache():
    # Clear Streamlit cache
    st.cache_data.clear()
    st.cache_resource.clear()
    print("Streamlit cache has been cleared.")

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

def main():
    choice = input("What would you like to do? Enter 'flush' to clear cache, 'delete' to remove files, or 'both' for both actions: ").strip().lower()

    # Files to be deleted
    files_to_delete = [
        './semantic_cache.json',
        './faiss_index.pkl',
    ]

    if choice == 'flush':
        flush_cache()
    elif choice == 'delete':
        delete_files(files_to_delete)
    elif choice == 'both':
        flush_cache()
        delete_files(files_to_delete)
    else:
        print("Invalid choice. Please enter 'flush', 'delete', or 'both'.")

if __name__ == "__main__":
    main()
