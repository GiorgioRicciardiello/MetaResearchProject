import pandas as pd
from config.config import config

if __name__ == '__main__':
    # Load the Excel file; update the filename if needed.
    df = pd.read_excel(config.get('embase_pp'))
    print(f'Number of unique papers to process: {df.Title.nunique()}')

    # Define inclusion keywords (using regex to match whole words).
    include_pattern = r'\b(questionnaire|survey|self-report)\b'

    # Define exclusion keywords to filter out studies using other data sources.
    exclude_pattern = r'\b(MRI|fMRI|functional MRI|PET|positron emission tomography|DTI|diffusion tensor imaging|MEG|magnetoencephalography|CT|computed tomography|neuroimaging|brain imaging|EEG|electroencephalography|ERP|actigraphy|wearable|accelerometer|sensor-based|electronic monitoring|biomarker|biomarkers|physiological|blood sample|genetic|genomics|DNA|epigenetics|biochemical|laboratory test|clinical assessment|biometric|cognitive task|behavioral task|reaction time|performance test|computerized assessment|smartphone tracking|app-based tracking|GPS tracking|objective measurement)\b'

    # Filter rows: include those that mention questionnaire-related terms and exclude those with any of the other terms.
    filtered_df = df[
        df['Abstract'].str.contains(include_pattern, case=False, na=False) &
        ~df['Abstract'].str.contains(exclude_pattern, case=False, na=False)
    ]
    print(f'Number of unique papers after processing {filtered_df}')
    # Optionally, print or save the filtered results.
    print(filtered_df)
    filtered_df.to_excel(config.get('embase_pp_filtered'))
    # To save the filtered results to a new Excel file, uncomment the next line:
    # filtered_df.to_excel('filtered_papers.xlsx', index=False)
