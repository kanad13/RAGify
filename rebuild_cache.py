# The pages/02-Chatbot.py file uses two types of caching:
	# 1. Streamlit's @st.cache_data for PDF processing
	# 2. A custom semantic cache for query results
# The code already has the ability to flush cache when the input file hash changes.
# This file is an extra "just-in-case" script to flush the cache manually.

# Running this script will flush all caches and rebuild indexes. This is what the code does
	# Clears Streamlit's cache_data and cache_resource.
	# Removes the 'semantic_cache.json' file.
	# Runs your main Streamlit script to rebuild all caches and indexes with fresh data.
# Run the file using python rebuild_caches.py
# This approach ensures that all caches are cleared and rebuilt with the latest data from the input files.

import subprocess
import streamlit as st
import os

def flush_caches_and_rebuild():
    # Clear Streamlit cache
    st.cache_data.clear()
    st.cache_resource.clear()

    # Remove semantic cache file
    if os.path.exists('semantic_cache.json'):
        os.remove('semantic_cache.json')

    print("All caches have been cleared.")

    # Rebuild indexes by running the main script
		#os.system('streamlit run Welcome.py')
    #subprocess.run(['streamlit', 'run', 'Welcome.py'], check=True)

if __name__ == "__main__":
    flush_caches_and_rebuild()
