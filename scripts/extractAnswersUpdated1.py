import re
import os
import csv
import spacy
from fuzzywuzzy import fuzz

# Load SpaCy medium model (supports word vectors)
nlp = spacy.load("en_core_web_md")

# Reference question and keywords
TARGET_QUESTION = "Where do you think you and your family are on this ladder?"
TARGET_KEYWORDS = ["where", "family", "ladder"]

def load_transcript(filepath):
    """Load transcript content from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def parse_transcript(text):
    """
    Parse transcript with inconsistent formatting.
    Handles missing IDs, timestamps, speakers, and cleans introductory phrases.
    """
    blocks = re.split(r'\n\s*\n', text.strip())
    transcript = []
    last_speaker = 'UNKNOWN'

    id_pattern = re.compile(r'^([a-zA-Z0-9\-]+)$')
    timestamp_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3})')
    # Speaker pattern includes 'child' as a possibility
    speaker_pattern = re.compile(r'^(interviewer|adolescent|parent|child)\s*:?\s*(.*)', re.IGNORECASE | re.DOTALL)

    for block in blocks:
        if not (current_block := block.strip()):
            continue
        uid, timestamp, content = 'id', '__:__:__:__ --> __:__:__:__', ''
        lines = current_block.split('\n')
        if len(lines) > 1 and id_pattern.match(lines[0].strip()):
            uid = lines.pop(0).strip()
            current_block = '\n'.join(lines)
        if ts_match := timestamp_pattern.search(current_block):
            timestamp, content_part = ts_match.group(1).replace(',', '.'), current_block[ts_match.end():].strip()
        else:
            content_part = current_block
        if speaker_match := speaker_pattern.match(content_part):
            speaker, content = speaker_match.group(1).upper(), speaker_match.group(2).strip()
            last_speaker = speaker
        else:
            speaker, content = last_speaker, content_part.strip()
        if not content: continue
        sentences = re.split(r'(?<=[.?!])\s+', content)
        cleaned_sentences = []
        for sentence in sentences:
            if match := re.search(r'could you tell\s+', sentence, re.IGNORECASE):
                modified_sentence = sentence[match.end():].strip()
                if modified_sentence.lower().startswith(('me ', 'us ')):
                    modified_sentence = modified_sentence[3:]
                cleaned_sentences.append(modified_sentence)
            else:
                cleaned_sentences.append(sentence)
        transcript.append({'id': uid, 'timestamp': timestamp, 'speaker': speaker, 'text': [s.strip() for s in cleaned_sentences if s.strip()]})
    return transcript

def combined_similarity(a, b):
    """Compute weighted similarity score using semantic and fuzzy matching."""
    doc_a, doc_b = nlp(a), nlp(b)
    semantic_sim = doc_a.similarity(doc_b)
    fuzzy_sim = fuzz.ratio(a.lower(), b.lower()) / 100
    return (semantic_sim * 0.6 + fuzzy_sim * 0.4)

def find_question_and_responses(transcript):
    """Find the best matching question and get responses that follow it."""
    best_score, best_index = -1, None
    for i, entry in enumerate(transcript):
        if entry['speaker'] == 'INTERVIEWER':
            for question in entry['text']:
                doc = nlp(question.lower())
                question_lemmas = {token.lemma_ for token in doc}
                matched_keyword_count = sum(1.5 if kw == 'family' else 1 for kw in TARGET_KEYWORDS if kw in question_lemmas)
                if matched_keyword_count >= 2:
                    score = matched_keyword_count + combined_similarity(question, TARGET_QUESTION)
                    if score > best_score:
                        best_score, best_index = score, i
    if best_index is not None:
        responses = []
        j = best_index + 1
        while j < len(transcript) and transcript[j]['speaker'] != 'INTERVIEWER':
            responses.append(transcript[j])
            j += 1
        return transcript[best_index], responses
    else:
        return None, []

def append_to_csv(output_filepath, data_row):
    """Appends a single row to a CSV file, creating it if it doesn't exist."""
    # The 'a' mode stands for append. `newline=''` is important to prevent blank rows.
    with open(output_filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data_row)

if __name__ == "__main__":
    data_folder = "data"
    output_csv_file = "ladder_question_responses.csv"

    # --- Setup CSV file with header ---
    # This creates the file and writes the header row once before processing files.
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["SessionId", "responseQ1", "participant"])
    
    # Process all txt files in the data folder and subfolders
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".txt"):
                filepath = os.path.join(root, file)
                print(f"\nProcessing file: {filepath}")

                transcript_text = load_transcript(filepath)
                if not transcript_text.strip():
                    print("SKIPPED: Empty transcript file.")
                    continue
                    
                transcript_entries = parse_transcript(transcript_text)
                if not transcript_entries:
                    print("SKIPPED: No valid transcript entries found.")
                    continue
                
                question_entry, responses = find_question_and_responses(transcript_entries)

                if question_entry and responses:
                    # --- Logic to extract data for CSV ---
                    
                    # 1. Extract SessionId from filename
                    session_id_match = re.search(r'([ap]_?\d+[a-z]?\s*,\s*[ap]\d+)', file, re.IGNORECASE)
                    session_id = session_id_match.group(1) if session_id_match else os.path.basename(filepath)

                    # 2. Combine all response text into a single string
                    full_response = " ".join(" ".join(res['text']) for res in responses).strip()

                    # 3. Determine participant type from speakers in the response
                    speakers = {res['speaker'] for res in responses}
                    is_parent = 'PARENT' in speakers
                    is_adolescent = 'ADOLESCENT' in speakers or 'CHILD' in speakers
                    
                    participant = 'unknown'
                    if is_parent and is_adolescent:
                        participant = 'both'
                    elif is_parent:
                        participant = 'parent'
                    elif is_adolescent:
                        participant = 'adolescent'

                    # 4. Append the extracted data to the CSV file
                    append_to_csv(output_csv_file, [session_id, full_response, participant])
                    
                    # --- Console Output ---
                    print(f" Match found for '{session_id}'. Result written to CSV.")
                    print(f"Matched Question: {' '.join(question_entry['text'])}")
                    print("Responses:")
                    for res in responses:
                        print(f"  [{res['speaker']}]: {' '.join(res['text'])}")
                else:
                    print(" No matching question and response found in this file.")
    
    print(f"\n Processing complete. All results saved to '{output_csv_file}'.")