import os
import time
import pandas as pd
import requests
from config.config import config
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import requests
from config.config import config


def hf_extract_abstract(page_text):
    """
    Uses a Hugging Face model (e.g., Llama-2) to extract an abstract from the provided page text.
    It sends a prompt instructing the model to extract a concise abstract.
    """
    prompt = (
            "Extract the abstract from the following page text. "
            "The abstract should be a concise summary of the paper's content. "
            "If the abstract is not clearly identified, return an empty string.\n\n"
            "Page Text:\n" + page_text + "\n\nAbstract:"
    )
    try:
        # Generate a response with a limit to prevent token overflows.
        result = hf_generator(prompt,
                              max_length=512,
                              truncation=True,
                              do_sample=False)
        generated_text = result[0]["generated_text"]
        # Remove the prompt portion from the result if it exists.
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text
    except Exception as e:
        print(f"Hugging Face extraction error: {e}")
        return ""


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/"
}


def get_page_text(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


def process_link(link):
    # First, get the full text from the URL
    page_text = get_page_text(link)
    return page_text

    # If the page text is empty, return an empty abstract
    if not page_text:
        return ""
    # Then, extract the abstract from the page text
    return hf_extract_abstract(page_text)

if __name__ == '__main__':
    # %%
    # --- Set up Selenium as before ---

    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    # chrome_options.binary_location = '/bin/chromium-browser'
    chrome_options.binary_location = config.get('chrome_options_binary_location')
    # service = Service('/bin/chromedriver')
    driver = webdriver.Chrome(chrome_options)

    # --- Set up the Hugging Face model ---
    # Replace with the model you prefer. Here we use Llama-2-7B-chat.
    # Note: Some models are gated and require authentication. Make sure to use your Hugging Face token if needed.
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_name = config.get('model_name') # meta-llama/Llama-2-7b-hf
    # model_name = "meta-llama/Meta-Llama-2-7b-chat-hf"
    hf_token = config.get('hf_token')  # personal token

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token
        )

    hf_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1  # device=0 for GPU, -1 for CPU
    )
    # %%

    # Load your DataFrame (ensure your CSV file is accessible)
    df = pd.read_csv(config.get('google_scholar_results'))

    # Apply the process_link function to each URL in the 'Link' column.
    df["complete_abstract"] = df["Link"].apply(process_link)

    print(df[["Link", "complete_abstract"]])


    # For testing, limit to a few rows
    # df = df.head(5)

    # Apply the extraction function to each URL in the 'Link' column.
    df["complete_sbtract"] = df["Link"].apply(hf_extract_abstract)

    # Save results to CSV.
    df.to_csv(config.get('google_scholar_abstracts'), index=False)

    # Clean up the Selenium driver.
    driver.quit()

    print("Abstract extraction complete. The DataFrame now has a 'complete_sbtract' column.")
