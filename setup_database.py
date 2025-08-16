# setup_database.py

#pandas is working with dataframes
import pandas as pd

# sentence transformer is for turning text into vector embeddings
from sentence_transformers import SentenceTransformer

# chromaDb is a vector database
import chromadb

# os and glob are to find the path
import os
import glob

# kagglehub to download dataset
import kagglehub
import subprocess
import requests
from tqdm import tqdm
import zipfile



# constants
DATASET_PATH = "myntra_fashion_data"
DB_PATH = "fashion_db"
COLLECTION_NAME = "products"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

def setup():
    """ 
    this function will performs the entire data setup process:
    1. download the dataset from kagglehub
    2. load csv into pandas.
    3. cleans and prepare the text data.
    4. inializes a persistent chromadb client
    5. creates embeddings for the product descriptions
    6. adds the embeddings and products info to the database
    """

    print( "---starting data setup---")

    # download and loading dataset


    print("\nstep 1. downloading and loading the dataset")

    # this downloads the dataset and saved into the dataset path

    # we will use our constant datapath to keep it clean

    # First, we ensure the data folder exists. If not, we download it.
    if not os.path.exists(DATASET_PATH):
        print(f"   '{DATASET_PATH}' not found. Downloading from Kaggle...")
        os.makedirs(DATASET_PATH)

        # The URL for the Kaggle API to download the dataset.
        dataset_slug = "hiteshsuthar101/myntra-fashion-product-dataset"
        url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_slug}"
        
        # The path where we'll save the downloaded zip file.
        zip_path = os.path.join(DATASET_PATH, "dataset.zip")

        try:
            # Step 1: Download the file using the 'requests' library.
            # 'stream=True' is important for downloading large files.
            response = requests.get(url, stream=True)
            response.raise_for_status() # This will stop if there's a download error (like 404).

            # Get the total size of the file for the progress bar.
            total_size_in_bytes = int(response.headers.get('content-length', 0))

            # Step 2: Create the progress bar using 'tqdm'.
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading")

            # Write the file to disk in small chunks and update the progress bar.
            with open(zip_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    progress_bar.update(len(chunk))
                    file.write(chunk)
            
            progress_bar.close() # Close the progress bar once finished.
            print("   -> Download complete.")

            # Step 3: Unzip the file.
            print("   -> Unzipping data...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATASET_PATH)
            
            # Step 4: Clean up by deleting the zip file.
            os.remove(zip_path)
            print("   -> Unzipping complete.")

        except Exception as e:
            print(f"\n   FATAL ERROR: Download failed. Please check your internet and Kaggle API key.")
            print(f"   Error details: {e}")
            return # Stop the script
        

    # Now that we know the folder exists, we find the CSV inside it.
    try:
        csv_path = glob.glob(os.path.join(DATASET_PATH, "**/*.csv"), recursive=True)[0]
        print(f"   Found CSV file at: {csv_path}")
    except IndexError:
        print(f"   FATAL ERROR: No CSV file found in '{DATASET_PATH}'.")
        return # STOP if no CSV is found
    

    df = pd.read_csv(csv_path)
    # Let's keep only a smaller, manageable subset for this example.
    # This is a realistic step for a junior dev to take to speed up testing.
    #df = df.head(10000) 
    print(f"   Loaded {len(df)} products into pandas.")


    # 2. prepare the text for embedding

    print("\nstep 2: preparing text data...")

    # fill any missing values with strings

    df['name'] = df['name'].fillna('').astype(str)
    df['brand'] = df['brand'].fillna('').astype(str)
    df['description'] = df['description'].fillna('').astype(str)

    df['full_description'] = df['brand'] + " " + df['name'] + ". " + df['description']


    print(f" created 'full_description' column for embedding.")

    # 3. initialize the database and the model

    print("\nstep 3: initializing embedding model and the database")

    # this line will loads the AI model that understands the text.

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # we create a persistent client.

    # it will save the database to a folder named 'fashion_db' in project directory.

    client = chromadb.PersistentClient(path= DB_PATH)

    # we get or create the collection. if it exists, we use it, if not then we create it.

    collection = client.get_or_create_collection(name= COLLECTION_NAME)
    print(f" database client initialized. collection '{COLLECTION_NAME} is ready.")

    # 4. generate embedding and add to db.

    print("\nstep 4: generating embedding and populating the database")

    # we will process in batches to be memory-friendly.
    batch_size = 4000
    num_items = len(df)

    for i in range(0, num_items, batch_size):

        #get the current batch of rows from our dataframe.
        batch_df = df.iloc[i : i + batch_size]

        # now get the text description of this batch.
        descriptions = batch_df['full_description'].tolist()

        # this is where the AI will work.
        embeddings = embedding_model.encode(descriptions)

        # prepare the metadata (the extra info we want to store)
        metadata = [
            {'name': row['name'], 'image_url': row['img'], 'description': row['full_description']}
            for  _, row in batch_df.iterrows()
        ]

        # prepare unique id for each item.

        ids = [str(idx) for idx in batch_df.index]

        # add this batch for our collection.
        collection.add(
            ids=ids,
            embeddings= embeddings.tolist(),
            metadatas= metadata
            
        )

        print(f" added batch {i//batch_size + 1}/{(num_items//batch_size) + 1} to the database.")

    print("\n--data setup complete--")

    print(f"total items in collection: {collection.count()}")

# this makes the script runnable from the command line.

if __name__ ==  "__main__":
    setup()
