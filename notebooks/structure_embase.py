"""
Emabse output the files as rows and not columns. Here we convert the files as columns
"""
from config.config import config
import pandas as pd

if __name__ == '__main__':
    df = pd.read_excel(config.get('embase_raw'))

    # Extract title and abstract pairs correctly
    titles = df[df.iloc[:, 1] == "TITLE"].iloc[:, 2].values
    abstracts = df[df.iloc[:, 1] == "ABSTRACT"].iloc[:, 2].values

    # Ensure both lists have the same length by truncating the longer one
    min_length = min(len(titles), len(abstracts))
    titles = titles[:min_length]
    abstracts = abstracts[:min_length]

    # Create DataFrame with extracted data
    df_titles_abstracts = pd.DataFrame({"Title": titles, "Abstract": abstracts})
    df_titles_abstracts.to_excel(config.get('embase_pp'))
    # Display the extracted data to the user
