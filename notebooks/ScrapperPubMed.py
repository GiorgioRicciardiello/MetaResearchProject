import requests
from bs4 import BeautifulSoup
import csv
import time
from typing import List, Dict

BASE_URL = "https://pubmed.ncbi.nlm.nih.gov"
# Your query as provided
QUERY = "(machine learning) AND (obstructive sleep apnea OR OSA) AND (questionnaire OR survey OR screening) AND (classification OR diagnosis)"

def fetch_search_results(query: str, max_pages: int = 1) -> List[Dict[str, str]]:
    """
    Fetches search results from PubMed for a given query over a specified number of pages.

    This function constructs the search URL for each page, sends a GET request,
    and then parses the returned HTML to extract the title and link of each paper.

    Args:
        query (str): The search query string.
        max_pages (int, optional): Number of pages of results to iterate over. Defaults to 1.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the keys:
            - 'title': The title of the paper.
            - 'link': The complete URL to the paper's page.
    """
    papers: List[Dict[str, str]] = []
    formatted_query = query.replace(" ", "+")
    for page in range(1, max_pages + 1):
        print(f"Fetching page {page}...")
        search_url = f"{BASE_URL}/?term={formatted_query}&page={page}"
        response = requests.get(search_url)
        if response.status_code != 200:
            print(f"Failed to retrieve page {page}: {response.status_code}")
            continue
        soup = BeautifulSoup(response.text, 'html.parser')
        # Each result is within a div with class "docsum-content"
        results = soup.find_all('div', class_='docsum-content')
        for result in results:
            title_link = result.find('a', class_='docsum-title')
            if title_link:
                title = title_link.get_text(strip=True)
                # The link is relative; combine it with BASE_URL.
                link = BASE_URL + title_link.get('href')
                papers.append({'title': title, 'link': link})
        # Be respectful to the server by pausing between pages.
        time.sleep(1)
    return papers


def fetch_abstract(paper_url: str) -> str:
    """
    Retrieves the abstract for a given paper URL from PubMed.

    This function sends a GET request to the provided paper URL,
    then parses the HTML to extract the abstract text found within a div
    with the class "abstract-content".

    Args:
        paper_url (str): The complete URL to the paper's page.

    Returns:
        str: The abstract text if found; otherwise, a message indicating unavailability.
    """
    time.sleep(1)
    response = requests.get(paper_url)
    if response.status_code != 200:
        return "Abstract not available"
    soup = BeautifulSoup(response.text, 'html.parser')
    abstract_div = soup.find('div', class_='abstract-content')
    if abstract_div:
        return abstract_div.get_text(strip=True)
    return "Abstract not available"


def save_to_csv(papers: List[Dict[str, str]], filename: str = 'pubmed_results.csv') -> None:
    """
    Saves a list of papers to a CSV file.

    Each paper should be a dictionary containing the keys:
    'title', 'abstract', and 'link'.

    Args:
        papers (List[Dict[str, str]]): List of papers to save.
        filename (str, optional): The filename for the CSV output. Defaults to 'pubmed_results.csv'.
    """
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'abstract', 'link']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for paper in papers:
            writer.writerow(paper)
    print(f"Results saved to {filename}")


def main() -> None:
    """
    Main function to execute the PubMed scraping workflow.

    - Prompts the user for the number of pages to scrape.
    - Fetches search results based on the global QUERY.
    - Iterates through each paper to fetch its abstract.
    - Saves the final data to a CSV file.
    """
    try:
        max_pages = int(input("Enter the number of pages to scrape: ").strip())
    except ValueError:
        print("Invalid number of pages. Exiting.")
        return

    print("Fetching search results...")
    papers = fetch_search_results(QUERY, max_pages)
    print(f"Found {len(papers)} papers. Fetching abstracts...")
    for i, paper in enumerate(papers, start=1):
        print(f"Fetching abstract for paper {i}...")
        paper['abstract'] = fetch_abstract(paper['link'])
    save_to_csv(papers)


if __name__ == '__main__':
    main()


    string = ""