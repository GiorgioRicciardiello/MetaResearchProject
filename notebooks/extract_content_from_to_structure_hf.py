"""
Using Hugging Face
"""
import os
import requests
import json
import PyPDF2


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


def extract_info_with_hf(text, prompt):
    """
    Uses Hugging Face's Inference API to extract key details from the paper text.
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
    full_prompt = f"{prompt}\n\nPaper text:\n{text[:2000]}"  # Adjust slicing if needed

    # Specify the Hugging Face model endpoint (change the model name as desired)
    API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    headers = {"Authorization": "Bearer YOUR_HF_API_TOKEN"}

    payload = {
        "inputs": full_prompt,
        "parameters": {"max_new_tokens": 300}  # Adjust token limit as needed
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        # Depending on the model's response format, adjust this extraction.
        generated_text = result[0]["generated_text"]
        extracted_data = json.loads(generated_text)
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
            extracted_info = extract_info_with_hf(text, custom_prompt)
            results[filename] = extracted_info

    # Print or save the results as JSON
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
