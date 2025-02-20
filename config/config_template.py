import pathlib
# Define root path
root_path = pathlib.Path(__file__).resolve().parents[1]
res_path  = root_path.joinpath('results')

config = {
    'search_query':'(machine learning) AND (obstructive sleep apnea OR OSA) AND (questionnaire OR survey OR screening) AND (classification OR diagnosis)',
    'SERPAPI_KEY': "",
    'chrome_options_binary_location': r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    'model_name': "meta-llama/Llama-2-7b-chat-hf",
    'hf_token': '',
    'google_scholar_results':  res_path.joinpath('google_scholar_tracking'),
    'google_scholar_abstracts': res_path.joinpath('google_scholar_output_with_abstracts'),

}