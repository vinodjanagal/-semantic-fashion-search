# app.py

# --- IMPORTS ---
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from bs4 import BeautifulSoup


# --- CONSTANTS ---
# These should match what you used in the setup script!
DB_PATH = "fashion_db"
COLLECTION_NAME = "products"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
RERANKING_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# load models and db (with caching)....
@st.cache_resource
def load_models():
    """loads the embeddings and the reranking models from the sentence transformers. """
    print("loading AI models....")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    reranking_model = CrossEncoder(RERANKING_MODEL)
    print("model loaded successfully")
    return embedding_model,reranking_model

@st.cache_resource
def load_collection():
    """loads the persistent chromadb collection from disk"""

    print("connecting to the database")

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name= COLLECTION_NAME)
    print("database collection successful")

    return collection

def clean_html(raw_html):
    """
    Takes a string with HTML tags and returns a clean, readable text string.
    """
    soup = BeautifulSoup(raw_html, 'html.parser')
    text = soup.get_text(separator='\n').strip()
    text = text.replace('<br>', '\n')
    return text

# search function
def perform_search(query,collection, embed_model, rerank_model,top_k = 20, top_n = 5):
    """
    Performs a semantic search with a reranking step.
    - top_k: How many initial results to retrieve from the database.
    - top_n: How many final results to show the user after reranking.
    """

    # 1. embed the users query.

    query_embedding = embed_model.encode(query)


    # 2. query the database for the top_k most similar items.

    initial_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results = top_k
    )

    # 3. rerank the initial reults for better accuracy.

    # cross encoder is slower but smarter. it looks at the query and the product description.
    # at the same time to give a more accurate relevance score.

    initial_product_info = initial_results['metadatas'][0]
    rerank_pairs = [[query, product['description']] for product in initial_product_info]

    if not rerank_pairs:
        return []
    
    scores = rerank_model.predict(rerank_pairs)

    # 4. combine scores with product info and sort to get the best results.

    scored_products =  sorted(zip(scores, initial_product_info), key=lambda x: x[0], reverse=True)

    # return the top_n final results

    final_results = [product for score, product in scored_products]
    return final_results[:top_n]



# streamlit user interface

def main():
    st.set_page_config(layout="wide")
    st.title("Semantic Search Fashion Engine")

    # load everything when the app starts

    embedding_model, reranking_model = load_models()
    fashion_collection = load_collection()


    st.markdown("You can enter a query to find fashion products based on their context, not just keywords.")
    st.markdown("_Examples: 'a professional but comfortable outfit for a long flight', 'something elegant for an art gallery opening', 'a rugged jacket for hiking'_")

    user_query = st.text_input("search for....")

    if st.button("Search"):
        if user_query:
            with st.spinner("Searching for the best matches...."):
                final_recommendations = perform_search(
                    query = user_query,
                    collection= fashion_collection,
                    embed_model= embedding_model,
                    rerank_model = reranking_model
                )

            st.header(f" Top 5 Recommendations for '{user_query}'")

            if not final_recommendations:
                st.warning("Sorry, I couldn't find any matching products")


            else:
                # We will loop through each recommendation and create a new row for it.
                for i, product in enumerate(final_recommendations):
                    
                    # Create a two-column layout for each product.
                    # The ratio [1, 3] gives the image 1 part of the space and the text 3 parts.
                    col1, col2 = st.columns([1, 3])
                    
                    # --- Column 1: The Image ---
                    with col1:
                        st.image(
                            product.get('image_url', ''), 
                            use_container_width='always', 
                            caption=f"Rank {i + 1}"
                        )

                    # --- Column 2: The Details ---

                    with col2:
                        # Display the product name.
                        st.subheader(product.get('name', 'N/A'))

                        # Get the raw description from our search results.
                        raw_description = product.get('description', '')
                        
                        # Use our new function to clean the HTML.
                        clean_description = clean_html(raw_description)

                        # Display the cleaned text directly using st.text.
                        # st.text is great for showing pre-formatted text with line breaks.
                        st.text(clean_description)


                    # Add a horizontal line to visually separate each product listing.
                    st.markdown("--------")


if __name__ == "__main__":
    main()




    
