import re
import os
import csv
import spacy
from fuzzywuzzy import fuzz

# Load SpaCy medium model (supports word vectors)
nlp = spacy.load("en_core_web_md")

# Reference question and keywords
Races= ["black","hispanic/latino","white",'asian']


def load_transcript(filepath):
    """Load transcript content from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# def parse_transcript(text):
#     """
#     Parse transcript with inconsistent formatting.
#     Handles missing IDs, timestamps, speakers, and cleans introductory phrases.
#     """
#     blocks = re.split(r'\n\s*\n', text.strip())
#     transcript = []
#     last_speaker = 'UNKNOWN'

#     id_pattern = re.compile(r'^([a-zA-Z0-9\-]+)$')
#     timestamp_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3})')
#     speaker_pattern = re.compile(r'^(interviewer|adolescent|parent|child)\s*:?\s*(.*)', re.IGNORECASE | re.DOTALL)

#     for block in blocks:
#         if not (current_block := block.strip()):
#             continue
#         uid, timestamp, content = 'id', '__:__:__:__ --> __:__:__:__', ''
#         lines = current_block.split('\n')
#         if len(lines) > 1 and id_pattern.match(lines[0].strip()):
#             uid = lines.pop(0).strip()
#             current_block = '\n'.join(lines)
#         if ts_match := timestamp_pattern.search(current_block):
#             timestamp, content_part = ts_match.group(1).replace(',', '.'), current_block[ts_match.end():].strip()
#         else:
#             content_part = current_block
#         if speaker_match := speaker_pattern.match(content_part):
#             speaker, content = speaker_match.group(1).upper(), speaker_match.group(2).strip()
#             last_speaker = speaker
#         else:
#             speaker, content = last_speaker, content_part.strip()
#         cleaned_sentences = content
#         if not content: continue
#         if speaker.lower() == 'interviewer':
#             sentences = re.split(r'(?<=[.?!])\s+', content)
#             cleaned_sentences = []
#             for sentence in sentences:
#                 if match := re.search(r'could you tell\s+', sentence, re.IGNORECASE):
#                     modified_sentence = sentence[match.end():].strip()
#                     if modified_sentence.lower().startswith(('me ', 'us ')):
#                         modified_sentence = modified_sentence[3:]
#                     cleaned_sentences.append(modified_sentence)
#                 else:
#                     cleaned_sentences.append(sentence)
#             transcript.append({'id': uid, 'timestamp': timestamp, 'speaker': speaker, 'text': [s.strip() for s in cleaned_sentences if s.strip()]})
#         else:
#             transcript.append({'id': uid, 'timestamp': timestamp, 'speaker': speaker, 'text': cleaned_sentences})
#     for transcripts in transcript:
#         print(f" {transcripts}\n")

#     return transcript


def parse_transcript(text):
    """
    Parse transcript with inconsistent formatting.
    This robust version identifies entries by their ID/Timestamp structure,
    not by blank lines, correctly handling all formats.
    """
    # This new regex finds all valid blocks by looking for the ID and Timestamp header.
    # It captures the ID, Timestamp, and all content until the next block's ID.
    block_pattern = re.compile(
        r'(^[a-zA-Z0-9\-]+$)\n'                                           # Group 1: The ID on its own line
        r'(^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}$)\n?' # Group 2: The Timestamp on the next line
        r'(.*?)'                                                            # Group 3: All content that follows (non-greedy)
        r'(?=\n^[a-zA-Z0-9\-]+$|\Z)',                                       # Stops when it sees the next ID or the end of the file
        re.MULTILINE | re.DOTALL
    )
    
    speaker_pattern = re.compile(r'^(interviewer|adolescent|parent|child)\s*:?\s*(.*)', re.IGNORECASE | re.DOTALL)
    
    transcript = []
    last_speaker = 'UNKNOWN'

    # Iterate through all matches found by the pattern
    for match in block_pattern.finditer(text):
        # Extract the parts of each matched block
        uid, timestamp_str, content_part = match.groups()
        
        timestamp = timestamp_str.replace(',', '.')
        content_part = content_part.strip()
        
        # --- Speaker and Content Logic ---
        # This part now operates on correctly separated blocks
        speaker_match = speaker_pattern.match(content_part)
        if speaker_match:
            speaker = speaker_match.group(1).upper()
            content = speaker_match.group(2).strip()
            last_speaker = speaker
        else:
            # If no speaker is found in the content, inherit the last one
            speaker = last_speaker
            content = content_part

        # --- Skip blocks with no dialogue content ---
        # This now correctly skips entries like 494 and 495
        if not content:
            continue

        # --- Sentence Cleaning Logic (Unchanged) ---
        sentences = re.split(r'(?<=[.?!])\s+', content)
        final_sentences = []
        
        if speaker == 'INTERVIEWER':
            for sentence in sentences:
                if m := re.search(r'could you tell\s+', sentence, re.IGNORECASE):
                    modified_sentence = sentence[m.end():].strip()
                    if modified_sentence.lower().startswith(('me ', 'us ')):
                        modified_sentence = modified_sentence[3:]
                    final_sentences.append(modified_sentence)
                else:
                    final_sentences.append(sentence)
        else:
            final_sentences = sentences
        
        # Final check to ensure we don't add entries with no text after cleaning
        processed_text = [s.strip() for s in final_sentences if s.strip()]
        if not processed_text:
            continue

        transcript.append({
            'id': uid,
            'timestamp': timestamp,
            'speaker': speaker,
            'text': processed_text,
        })
    # for transcript_entry in transcript:
    #     print(transcript_entry)
    return transcript

def combined_similarity(a, b):
    """Compute weighted similarity score using semantic and fuzzy matching."""
    doc_a, doc_b = nlp(a), nlp(b)
    semantic_sim = doc_a.similarity(doc_b)
    fuzzy_sim = fuzz.ratio(a.lower(), b.lower()) / 100
    return (semantic_sim * 0.6 + fuzzy_sim * 0.4)

# UPDATED: Replaced the old function with the new one
def find_why_question(transcript, current_index):
    while current_index < len(transcript):
        if current_index < len(transcript) and transcript[current_index]['speaker'].lower() == 'interviewer':
            interviewer_entry = transcript[current_index]
            # Check if any text in this entry contains "why"
            contains_why = any('why' in text.lower() for text in interviewer_entry['text'])
            current_index += 1
            if contains_why:
                return transcript[current_index-1]['text'], current_index
        else:
            current_index += 1
    return [], current_index

def find_rating_question_answers(transcript,word):
    TARGET_QUESTION = f"Where do you think {word} are on this ladder?"
    TARGET_KEYWORDS = ["where", word, "ladder"]

    best_score, best_index, best_question = -1, None, None
    for i, entry in enumerate(transcript):
        if entry['speaker'].lower() == 'interviewer':
            for question in entry['text']:
                doc = nlp(question.lower())
                question_lemmas = {token.lemma_ for token in doc}
                matched_keyword_count = sum(1.5 if kw == word else 1 for kw in TARGET_KEYWORDS if kw in question_lemmas)
                if matched_keyword_count >= 1.5:
                    score = matched_keyword_count + combined_similarity(question, TARGET_QUESTION)
                    if score > best_score:
                        best_score, best_index, best_question = score, i, question
    response_rating = []
    if best_index is None:
        return None, None, response_rating
    current_index = best_index + 1
    while current_index < len(transcript) and transcript[current_index]['speaker'].lower() != 'interviewer':
        response_rating.append(transcript[current_index]['text'])
        current_index += 1

    return best_index, best_question, response_rating

def find_question_responses_and_reason(transcript):
    """
    Find the best matching question, get the initial responses, 
    and capture the subsequent "reason" exchange.
    """
    best_index, best_question, response_rating = find_rating_question_answers(transcript, "family")

    if best_index is not None:
        question_entry = best_question
        reason_question = []
        reason_phrases = []
        current_index = best_index + 1
        while current_index < len(transcript) and transcript[current_index]['speaker'].lower() != 'interviewer':
            response_rating.append(transcript[current_index]['text'])
            current_index += 1
        why_question_output=find_why_question(transcript, current_index)
        why_question = None
        if why_question_output:
            why_question, current_index = why_question_output
        reason_question.append(why_question)
        
            
        while current_index < len(transcript) and transcript[current_index]['speaker'].lower() != 'interviewer':
            reason_phrases.append(transcript[current_index]['text'])
            current_index += 1

        return question_entry, response_rating, reason_question, reason_phrases

    return None, [], [], []

def append_to_csv(output_filepath, data_row):
    """Appends a single row to a CSV file."""
    with open(output_filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data_row)

if __name__ == "__main__":
    data_folder = "data"
    output_csv_file = "ladder_question_responses.csv"

    # UPDATED: Added the "reason" column to the CSV header
    race_columns = [f"response_{race.replace('/', '_or_')}" for race in Races]
    header = ["SessionId", "responseQ1_Family", "participant", "reason_Family"] + race_columns
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)    
        writer.writerow(header)

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
                
                # UPDATED: Unpack the new `reason_text` variable from the function call
                question_entry, responses, reason_questions, reason_phrases = find_question_responses_and_reason(transcript_entries)
              
                session_id_match = re.search(r'([ap]_?\d+[a-z]?\s*,\s*[ap]\d+)', file, re.IGNORECASE)
                session_id = session_id_match.group(1) if session_id_match else os.path.basename(filepath)
                if question_entry:
                    full_response = " ".join(" ".join(res) for res in responses).strip()
                    print(f"  Matched Question: {question_entry}")
                    print(f"  Initial Response: {full_response}")
                    if reason_questions and reason_phrases:
                        full_reason = " ".join(" ".join(reason) for reason in reason_phrases).strip()
                        full_reason_question = " ".join(" ".join(reason_question) for reason_question in reason_questions).strip()
                        print(f"  Extracted Reason: Interviewer:{full_reason_question}\n Response:{full_reason}")
                    else:
                        print("  (No subsequent reason exchange was found)")
                else:
                    print(" No matching question and response found in this file.")

                # UPDATED: Print the extracted reason for verification
                    
                filename_lower = file.lower()
                participant = 'unknown'
                if 'joint' in filename_lower:
                    participant = 'joint'
                elif 'adolescent' in filename_lower:
                    participant = 'adolescent'
                elif 'parent' in filename_lower:
                    participant = 'parent'
                    
                all_race_responses = []
                for race in Races:
                        # Find the rating question and response for the current race
                    best_index_race, best_question_race, response_rating_race = find_rating_question_answers(transcript_entries, race)
                        # Format the response into a single string
                    full_race_response = " ".join(" ".join(res) for res in response_rating_race).strip()
                    print(f"  Race: {race}, Best Question: {best_question_race}, Response Rating: {response_rating_race}")
                    all_race_responses.append(full_race_response)

                    # --- UPDATED: Combine all data and write to CSV ---
                data_row = [session_id, full_response, participant, full_reason] + all_race_responses
                append_to_csv(output_csv_file, data_row)                    
                    # --- Console Output ---
                    
                print(f"\n✅ Processing complete. All results saved to '{output_csv_file}'.")

# if __name__ == "__main__":
#     data_folder = "data"x
#     output_csv_file = "ladder_question_responses.csv"

#     # UPDATED: Added the "reason" column to the CSV header
#     with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["SessionId", "responseQ1", "participant", "reason"])
#                 filepath="data/DE-IDENTIFIED Wave 1 Transcripts/Joint Parent-Child Interviews/de-identified transcripts a1011,p101 - joint interview.txt"
            
#     # for root, _, files in os.walk(data_folder):
#     #     for file in files:
#             # if file.endswith(".txt"):
#                 # filepath = os.path.join(root, file)
#                 print(f"\nProcessing file: {filepath}")

#                 transcript_text = load_transcript(filepath)
#                 if not transcript_text.strip():
#                     print("SKIPPED: Empty transcript file.")
#                     # continue
                    
#                 transcript_entries = parse_transcript(transcript_text)
#                 if not transcript_entries:
#                     print("SKIPPED: No valid transcript entries found.")
#                     # continue
                
#                 # UPDATED: Unpack the new `reason_text` variable from the function call
#                 question_entry, responses, reason_questions, reason_phrases = find_question_responses_and_reason(transcript_entries)
                
                

#                 if question_entry and responses:
#                     # --- Logic to extract data for CSV ---
                    
#                     session_id_match = re.search(r'([ap]_?\d+[a-z]?\s*,\s*[ap]\d+)', filepath, re.IGNORECASE)
#                     session_id = session_id_match.group(1) if session_id_match else os.path.basename(filepath)
#                     # print(reason_phrases)

#                     full_response = " ".join(" ".join(res) for res in responses).strip()
#                     full_reason = " ".join(" ".join(reason) for reason in reason_phrases).strip()
#                     full_reason_question = " ".join(" ".join(reason_question) for reason_question in reason_questions).strip()
#                     filename_lower = filepath.lower()
#                     participant = 'unknown'
#                     if 'joint' in filename_lower:
#                         participant = 'joint'
#                     elif 'adolescent' in filename_lower:
#                         participant = 'adolescent'
#                     elif 'parent' in filename_lower:
#                         participant = 'parent'
                        
                   
#                     # UPDATED: Add the `reason_text` to the data row for the CSV
#                     # append_to_csv(output_csv_file, [session_id, full_response, participant, full_reason])
                    
#                     # --- Console Output ---
#                     print(f" Match found for '{session_id}'. Result written to CSV.")
#                     print(f"  Matched Question: {question_entry}")
#                     print(f"  Initial Response: {full_response}")
                    
#                     # UPDATED: Print the extracted reason for verification
#                     if reason_questions and reason_phrases:
#                         print(f"  Extracted Reason: Interviewer:{full_reason_question} Response:{full_reason}")
#                     else:
#                         print("  (No subsequent reason exchange was found)")
                    
#                     for race in Races:
#                     # Logic to handle different races
#                         best_index, best_question, response_rating = find_rating_question_answers(transcript_entries, race)
#                         print(f"  Race: {race}, Best Question: {best_question}, Response Rating: {response_rating}")

                    
#                 else:
#                     print(" No matching question and response found in this file.")
    
#     print(f"\n✅ Processing complete. All results saved to '{output_csv_file}'.")
    
    