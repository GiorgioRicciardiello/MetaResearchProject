import os
import openai
import PyPDF2
import json
from config.config import config

# Set your OpenAI API key
openai.api_key = config.get('OPENAI_KEY')

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file using PyPDF2."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_info_with_openai(text, prompt):
    """
    Uses OpenAI's ChatCompletion API to extract key details from the paper text.
    The provided 'prompt' should include instructions for extracting:
      - Total sample size
      - Training samples
      - Test samples
      - Sensitivity and specificity metrics for each AHI threshold (e.g., AHI>5, AHI>15, AHI>30),
        along with any definitions or notes regarding the thresholds.
    If any detail is not present, the output should be 'nan'.

    The prompt is concatenated with the first 2000 characters of the paper text to avoid token limits.
    Returns a dictionary with the extracted information.
    """
    # Combine the custom prompt with the paper text
    full_prompt = (
        f"{prompt}\n\nPaper text:\n{text[:2000]}"  # Adjust text slicing as needed
    )

    try:

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # or choose another model
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that extracts quantitative information from scientific texts."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        extracted = response.choices[0].message.content.strip()

        # # Parse the assistant's reply
        # reply = response.choices[0].message.content.strip()
        # extracted_data = json.loads(reply)
    except Exception as e:
        print("Error extracting data:", e)
        extracted_data = {}

    return extracted_data


def main():
    # Define your custom extraction prompt
    custom_prompt = (
        "Extract the following details from the paper text provided:\n"
        "- Total sample size\n"
        "- Training samples\n"
        "- Test samples\n"
        "- Sensitivity and specificity values for each AHI threshold defined in the paper. "
        "For example, if the paper provides metrics for AHI>5, AHI>15, or AHI>30, extract these. "
        "Also include any notes on how the paper defines the AHI thresholds. "
        "If any of these details are not present, output 'nan' for that value.\n\n"
        "Provide the output in JSON format with the following keys:\n"
        "  'total_sample_size': (value or 'nan')\n"
        "  'training_samples': (value or 'nan')\n"
        "  'test_samples': (value or 'nan')\n"
        "  'AHI_metrics': an array of objects, each with the keys:\n"
        "      'AHI_threshold': the threshold (e.g., 'AHI>5')\n"
        "      'sensitivity': sensitivity value or 'nan'\n"
        "      'specificity': specificity value or 'nan'\n"
        "      'definition': notes on the AHI definition or 'nan' if not available"
    )

    # Path to the directory containing PDF files
    pdf_dir = "path/to/your/pdf/folder"  # Update with your directory path
    results = {}

    # Iterate over all PDF files in the directory
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"Processing {filename}...")
            text = extract_text_from_pdf(pdf_path)
            extracted_info = extract_info_with_openai(text, custom_prompt)
            results[filename] = extracted_info

    # Print or save the results as JSON
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
