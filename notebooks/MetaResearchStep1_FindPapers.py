import requests
import pandas as pd
import spacy
from fuzzywuzzy import fuzz
import time
from config.config import config

if __name__ == '__main__':
    # Initialize NLP model
    nlp = spacy.load("en_core_web_sm")

    # Set SerpAPI key (Replace with your actual key)
    SERPAPI_KEY = config.get("SERPAPI_KEY")
    # Define search query for Google Scholar
    search_query = config.get("search_query")


    def search_google_scholar(query, max_pages=5):
        """Search Google Scholar using SerpAPI and return all articles across multiple pages."""
        all_articles = []
        start_index = 0  # Google Scholar uses pagination with start indices

        for page in range(max_pages):
            print(f"🔍 Searching page {page + 1}...")
            params = {
                "engine": "google_scholar",
                "q": query,
                "num": 10,  # Number of results per page (Google Scholar default)
                "start": start_index,  # Controls pagination
                "api_key": SERPAPI_KEY
            }

            response = requests.get("https://serpapi.com/search", params=params)
            results = response.json()

            # Extract articles from results
            articles = results.get("organic_results", [])
            if not articles:
                print("⚠️ No more results found.")
                break  # Stop searching if no more results

            all_articles.extend(articles)
            start_index += 10  # Move to the next page
            time.sleep(2)  # Prevent rate limiting

        return all_articles


    def is_relevant(abstract, inclusion_criteria, exclusion_criteria):
        """Determine if an abstract meets inclusion criteria and does not meet exclusion criteria."""
        doc = nlp(abstract.lower())

        # Check if at least one inclusion criterion is present
        inclusion_check = any(any(keyword in token.text for token in doc) for keyword in inclusion_criteria)

        # Check if any exclusion criterion is present
        exclusion_check = any(any(keyword in token.text for token in doc) for keyword in exclusion_criteria)

        return inclusion_check and not exclusion_check


    def remove_duplicates(df):
        """Remove duplicate articles based on title similarity."""
        unique_titles = []
        filtered_df = []

        for _, row in df.iterrows():
            title = row["Title"]
            if not any(fuzz.ratio(title, t) > 90 for t in unique_titles):
                unique_titles.append(title)
                filtered_df.append(row)

        return pd.DataFrame(filtered_df)


    # Perform the search on Google Scholar
    articles = search_google_scholar(search_query, max_pages=50)

    # Extract metadata
    data = []
    for article in articles:
        title = article.get("title", "No title available")
        abstract = article.get("snippet", "No abstract available")  # Google Scholar does not provide full abstracts
        link = article.get("link", "No link available")

        data.append([title, abstract, link])

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Title", "Abstract", "Link"])

    # Print total number of search results
    print(f"🔍 Google Scholar search returned {len(df)} results.")

    # Define inclusion and exclusion criteria
    inclusion_criteria = [
        "machine learning", "obstructive sleep apnea", "questionnaire", "classification accuracy"
    ]
    exclusion_criteria = [
        "physiological signals", "ECG", "SpO2", "logistic regression", "review", "commentary", "case report"
    ]

    # Filter articles based on relevance
    df["Relevant"] = df["Abstract"].apply(lambda x: is_relevant(str(x), inclusion_criteria, exclusion_criteria))

    # Remove duplicates
    df_cleaned = remove_duplicates(df[df["Relevant"]])

    # Add selection column
    df_cleaned["Selected"] = False  # Default: Not Selected

    # Save to CSV
    df_cleaned.to_csv(config.get('google_scholar_results'), index=False)

    print(f"✅ Search and filtering completed. {len(df_cleaned)} relevant studies saved.")
