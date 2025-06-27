import os
import glob

GOOD_CALLS_DIR = os.path.join(os.path.dirname(__file__), '../data/good_calls')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../data/good_calls_cleaned')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)

def process_all_files():
    txt_files = glob.glob(os.path.join(GOOD_CALLS_DIR, '*.txt'))
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()
        cleaned = clean_text(raw)
        base = os.path.basename(file_path)
        out_path = os.path.join(OUTPUT_DIR, base)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f'Processed: {base}')

if __name__ == '__main__':
    process_all_files()
    print('All good call transcripts have been cleaned and saved to good_calls/.') 