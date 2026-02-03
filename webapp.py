
import os
import sys
import threading
import webbrowser
import tempfile
import time
import uuid
import json
import re
import difflib
import traceback
import math
from datetime import timedelta, datetime
from flask import Flask, request, send_file, render_template_string, jsonify, after_this_request

# Check Libraries
try:
    # Attempt to add user site packages explicitly
    import site
    import sys
    try:
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.append(user_site)
    except: pass
    
    from pythainlp.tokenize import word_tokenize
    from pythainlp.util import dict_trie
    try:
        from pythainlp.corpus import thai_words
    except ImportError:
        from pythainlp import thai_words
        
    PYTHAINLP_AVAILABLE = True
    THAI_DICT = set(thai_words())
    print("PyThaiNLP Loaded Successfully!")
except ImportError as e:
    PYTHAINLP_AVAILABLE = False
    THAI_DICT = set()
    print(f"WARNING: PyThaiNLP import failed: {e}")

try:
    import requests
    import time  # Ensure time is imported
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

if __name__ == "__main__":
    print("-" * 50)
    print(f"DEBUG STATUS:")
    print(f"PyThaiNLP Available: {PYTHAINLP_AVAILABLE}")
    print(f"Requests Available: {REQUESTS_AVAILABLE}")
    print("-" * 50)

app = Flask(__name__)
model = None
jobs = {}

# --- Config Loading ---
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    default_config = {
        'gemini_api_key': '',
        'gemini_model': 'gemini-2.5-flash',  # Default model (Updated to 2.5 Flash Stable)
        'use_dictionary_filter': True,
        'use_gemini_correction': False
    }
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            return {**default_config, **cfg}
    except:
        return default_config

CONFIG = load_config()

# --- Progress Capture Helper ---
class ProgressCapture:
    """
    Captures tqdm output from stderr to estimate progress percentage.
    """
    def __init__(self, job_id):
        self.job_id = job_id
        self.original_stderr = sys.stderr
        self.original_stdout = sys.stdout
        self.buffer = ""

    def write(self, data):
        # We write to original streams so user still sees them in terminal
        if sys.stderr == self:
             self.original_stderr.write(data)
        elif sys.stdout == self:
             self.original_stdout.write(data)
             
        self.buffer += data
        try:
            # Match percentage for progress bar
            match = re.search(r'\s(\d+)%\|', data)
            if match:
                pct = int(match.group(1))
                if jobs.get(self.job_id):
                    jobs[self.job_id]['message'] = f"AI กำลังถอดความ... {pct}%"
                    jobs[self.job_id]['progress'] = pct
        except:
            pass

    def flush(self):
        self.original_stderr.flush()
        self.original_stdout.flush()

    def __enter__(self):
        sys.stderr = self
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stderr = self.original_stderr
        sys.stdout = self.original_stdout

# --- Logic ---

def format_time(td):
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    microseconds = td.microseconds
    return f"{hours:02}:{minutes:02}:{seconds:02},{microseconds // 1000:03}"

def parse_time_str(t_str):
    try:
        # Normalize: replace , and : (in ms position) with .
        t_str = t_str.strip().replace(',', '.')
        
        # Split by : first
        parts = t_str.split(':')
        
        if len(parts) == 3:
            # Could be HH:MM:SS.mmm or MM:SS:mmm (weird Gemini format)
            # Check if last part looks like SS.mmm or just mmm
            last_part = parts[2]
            if '.' in last_part:
                # Normal format: HH:MM:SS.mmm
                ss, ms = last_part.split('.', 1)
                h, m, s = int(parts[0]), int(parts[1]), int(ss)
                ms_int = int(ms.ljust(6, '0')[:6])  # pad/truncate to microseconds
            else:
                # Weird format: MM:SS:mmm (Gemini sometimes does this)
                # Treat as: 00:MM:SS.mmm
                h = 0
                m, s = int(parts[0]), int(parts[1])
                ms_int = int(parts[2].ljust(6, '0')[:6])
        elif len(parts) == 2:
            # MM:SS.mmm format
            h = 0
            m = int(parts[0])
            last_part = parts[1]
            if '.' in last_part:
                ss, ms = last_part.split('.', 1)
                s = int(ss)
                ms_int = int(ms.ljust(6, '0')[:6])
            else:
                s = int(last_part)
                ms_int = 0
        else:
            raise ValueError(f"Unknown format: {t_str}")
        
        return timedelta(hours=h, minutes=m, seconds=s, microseconds=ms_int)
    except Exception as e:
        print(f"Time parse error: {t_str} -> {e}")
        return None


def extract_text_from_srt(content):
    lines = content.splitlines()
    text_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.isdigit(): continue
        if "-->" in line: continue
        text_lines.append(line)
    return " ".join(text_lines)

def split_text_logic(text, start, end, max_chars, min_chars):
    chunks = []
    
    if PYTHAINLP_AVAILABLE:
        # Intelligent splitting with PyThaiNLP + Heuristics
        all_tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
        all_tokens = [t for t in all_tokens if t.strip()]
        
        # Sticky Particles (Glue to Previous)
        STICKY_END = {'ครับ', 'ค่ะ', 'คะ', 'นะ', 'จ้ะ', 'จ๋า', 'ไหม', 'มั้ย', 'หรอ', 'เหรอ', 'หรือเปล่า', 'เบย', 'เลย', 'ด้วย', 'เอง'}
        # Dangling Connectors (Glue to Next)
        DANGLING_START = {'ที่', 'ซึ่ง', 'อัน', 'ใน', 'ของ', 'กับ', 'แก่', 'แด่', 'ต่อ', 'เพื่อ', 'โดย', 'และ', 'หรือ', 'แต่', 'ถ้า', 'หาก', 'เมื่อ', 'จน', 'กระทั่ง'}
        
        total_tokens_chars = sum(len(t) for t in all_tokens)
        committed_chars = 0
        current_chunk = ""
        
        for i, token in enumerate(all_tokens):
            if not current_chunk:
                current_chunk = token
                continue
            
            # Dynamic Target Calculation
            # Calculate how much text is left including current chunk
            remaining_total = total_tokens_chars - committed_chars
            if remaining_total <= 0: remaining_total = 1 # Avoid div zero
            
            chunks_needed = math.ceil(remaining_total / max_chars)
            # Ideal target length for this chunk to keep balance
            target_len = remaining_total / chunks_needed if chunks_needed > 0 else max_chars
            
            # Allow a small overflow over target (e.g. +10%) before soft breaking, 
            # unless it hits max_chars (Hard Limit)
            soft_limit = min(max_chars, target_len * 1.2) # 20% flex for target
            
            current_len = len(current_chunk)
            token_len = len(token)
            
            will_overflow_hard = (current_len + token_len) > max_chars
            will_overflow_soft = (current_len + token_len) > soft_limit
            
            should_split = False
            
            # Heuristic 1: Force stickiness for small particles (Overrides Soft Limit, respects Hard Limit if possible)
            is_sticky = token in STICKY_END
            if is_sticky and not will_overflow_hard:
                current_chunk += token
                continue
                
            if will_overflow_hard:
                should_split = True
            elif will_overflow_soft:
                # If soft limit reached, we WANT to split, unless current token is a "Dangler" that should start a new line anyway?
                # Actually if it's a dangler, splitting here is GOOD.
                # If it is NOT a dangler, splitting here is okay too (neutral).
                should_split = True
            
            if should_split:
                 # Heuristic 2: Check for dangling Connector at end of current_chunk
                shifted = False
                for dangler in DANGLING_START:
                    if current_chunk.endswith(dangler) and len(current_chunk) > len(dangler):
                         prev_part = current_chunk[:-len(dangler)]
                         chunks.append(prev_part)
                         committed_chars += len(prev_part)
                         current_chunk = dangler + token
                         shifted = True
                         break
                         
                if not shifted:
                    chunks.append(current_chunk)
                    committed_chars += len(current_chunk)
                    current_chunk = token
            else:
                current_chunk += token
                
        if current_chunk:
            chunks.append(current_chunk)
    else:
        # Fallback: Character-based splitting
        current_chunk = ""
        words = text.split(' ')
        if len(words) < len(text) / 5: 
            desc = text
            while len(desc) > max_chars:
                 chunks.append(desc[:max_chars])
                 desc = desc[max_chars:]
            if desc: chunks.append(desc)
        else:
             for word in words:
                if not current_chunk:
                    current_chunk = word
                else:
                    if len(current_chunk) + len(word) + 1 > max_chars:
                        chunks.append(current_chunk)
                        current_chunk = word
                    else:
                        current_chunk += " " + word
             if current_chunk: chunks.append(current_chunk)

    if not chunks:
        return []

    total_duration = (end - start).total_seconds()
    total_chars = sum(len(c) for c in chunks)
    current_start = start
    result_subs = []
    
    for chunk in chunks:
        if total_chars > 0:
             duration_fraction = len(chunk) / total_chars
        else:
             duration_fraction = 1.0 / len(chunks)
             
        chunk_duration_sec = duration_fraction * total_duration
        chunk_end = current_start + timedelta(seconds=chunk_duration_sec)
        
        result_subs.append({
            "start": format_time(current_start),
            "end": format_time(chunk_end),
            "text": chunk
        })
        current_start = chunk_end
        
    return result_subs

def is_valid_thai_word(word):
    """Check if word exists in Thai dictionary."""
    if not PYTHAINLP_AVAILABLE or not word:
        return True
    # Allow numbers, punctuation, spaces
    if word.isdigit() or len(word) <= 1:
        return True
    # Check against Thai dictionary
    return word in THAI_DICT

def filter_with_dictionary(text, ref_text=None):
    """Filter out non-Thai words and replace with closest match from ref or dictionary."""
    if not PYTHAINLP_AVAILABLE or not CONFIG.get('use_dictionary_filter', True):
        return text
    
    tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
    ref_vocab = set()
    if ref_text:
        ref_tokens = word_tokenize(ref_text, engine='newmm', keep_whitespace=False)
        ref_vocab = set(ref_tokens)
    
    result = []
    for token in tokens:
        token_clean = token.strip()
        if not token_clean:
            continue
        
        # Priority: 1. In reference, 2. In Thai dict, 3. Find closest match
        if token_clean in ref_vocab or is_valid_thai_word(token_clean):
            result.append(token_clean)
        else:
            # Try to find similar word in reference first
            if ref_vocab:
                matches = difflib.get_close_matches(token_clean, list(ref_vocab), n=1, cutoff=0.6)
                if matches:
                    result.append(matches[0])
                    continue
            # Skip the word if no match (it's likely garbage)
            # Or keep it but mark for review
            result.append(token_clean)  # Keep for now
    
    return ''.join(result)

def estimate_tokens(text):
    # Rough estimate: 1 Thai char ~ 0.5-0.8 tokens depending on complexity
    # Gemini uses sentencepiece, but for safety lets assume 1 char = 1 token for worst case
    return len(text)

def call_gemini_api(prompt, api_key):
    """Internal helper to call Gemini API"""
    # Use model from config, default to 2.0-flash if not set
    model_name = CONFIG.get('gemini_model', 'gemini-2.0-flash')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    print(f"[Gemini Batch] Using Model: {model_name}") # Debug Log
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096}
    }
    
    est_input_tokens = estimate_tokens(prompt)
    print(f"[Gemini Batch] Est. Input Tokens: ~{est_input_tokens}")
    
    # Retry logic for Rate Limit (429) - More aggressive backoff
    # Free Tier Limit: 15 RPM (Requests Per Minute) -> 1 req every 4 seconds
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                corrected = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                
                # Check output usage
                usage = data.get('usageMetadata', {})
                print(f"[Gemini Batch] Usage: {usage}")
                
                if corrected:
                    return corrected.strip()
            elif resp.status_code == 429:
                wait_time = (attempt + 1) * 10  # Backoff: 10s, 20s, 30s (Slower is better for 429)
                print(f"[Gemini Batch] Rate Limit (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"[Gemini Batch] API Error: {resp.text[:200]}")
                break
        except Exception as e:
            print(f"[Gemini Batch] Exception: {e}")
            break
            
    return None

def correct_batch_with_gemini(text_list, ref_text=None):
    """Correct a batch of lines using Gemini."""
    api_key = CONFIG.get('gemini_api_key', '')
    if not api_key or not REQUESTS_AVAILABLE:
        return text_list

    # Prepare batch prompt
    lines_str = "\n".join([(f"Line{i+1}: {t}") for i, t in enumerate(text_list)])
    
    prompt = f"""คุณคือผู้เชี่ยวชาญภาษาไทย กรุณาแก้ไขข้อความต่อไปนี้ทีละบรรทัด:
- แก้คำผิด คำทับศัพท์ ให้ถูกต้องตามหลักภาษา
- [สำคัญ] ห้ามรวมบรรทัด หรือแยกบรรทัดใหม่ จำนวนบรรทัดต้องเท่าเดิม
- ส่งคืนเฉพาะข้อความที่แก้แล้ว เรียงตามลำดับ Line1, Line2... 
- ไม่ต้องใส่คำว่า "LineX:" นำหน้าในคำตอบ ให้ตอบข้อความมาเลย

ข้อความต้นฉบับ:
{lines_str}
"""
    if ref_text:
        prompt += f"\nบทบริบทอ้างอิง: {ref_text[:1000]}"

    print(f"[Gemini Batch] Sending batch of {len(text_list)} lines...")
    result = call_gemini_api(prompt, api_key)
    
    if result:
        corrected_lines = [l.strip() for l in result.split('\n') if l.strip()]
        # Verify length match
        if len(corrected_lines) == len(text_list):
            print(f"[Gemini Batch] Success! Updated {len(corrected_lines)} lines.")
            return corrected_lines
        else:
            print(f"[Gemini Batch] Length mismatch (In: {len(text_list)}, Out: {len(corrected_lines)}). Returning original.")
            # Fallback: try to match best effort or return original
            return text_list
            
    return text_list

def correct_text_with_ref(text, ref_text, run_gemini=True):
    # Step 1 & 2: Dictionary and Ref Matching (Require PyThaiNLP)
    if PYTHAINLP_AVAILABLE:
        text = filter_with_dictionary(text, ref_text)
        if ref_text:
            ref_tokens = word_tokenize(ref_text, engine='newmm', keep_whitespace=False)
            ref_vocab = set(ref_tokens)
            ref_words_list = list(ref_vocab)
            target_tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
            corrected_tokens = []
            for token in target_tokens:
                token_clean = token.strip()
                if not token_clean:
                    corrected_tokens.append(token)
                    continue
                if token_clean in ref_vocab:
                    corrected_tokens.append(token)
                    continue
                matches = difflib.get_close_matches(token_clean, ref_words_list, n=1, cutoff=0.7)
                if matches:
                    corrected_tokens.append(matches[0])
                else:
                    corrected_tokens.append(token)
            text = ''.join(corrected_tokens)
    
    # Step 3: Gemini correction (single line mode, if enabled)
    if run_gemini:
        # Compatibility wrapper for single line
        res = correct_batch_with_gemini([text], ref_text)
        return res[0] if res else text
    
    return text

def translate_batch_with_gemini(text_list, ref_text=None, api_key=None):
    """Translate Thai lines to English using Gemini."""
    if not api_key: return [""] * len(text_list)
    
    # Prepare batch
    lines_str = "\n".join([(f"Line{i+1}: {t}") for i, t in enumerate(text_list)])
    
    prompt = f"""You are a professional subtitle translator. Translate the following Thai subtitle lines to English.
Rules:
1. Maintain the exact same number of lines.
2. Keep the meaning accurate but natural for English speakers.
3. Keep it concise (subtitle style).
4. Return ONLY the translated lines, one per line.

Original Thai:
{lines_str}
"""
    if ref_text: prompt += f"\nContext/Ref: {ref_text[:500]}"
    
    print(f"[Gemini Translator] Translating {len(text_list)} lines...")
    result = call_gemini_api(prompt, api_key)
    
    if result:
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        # Strip "LineX:" key if Gemini adds it (it shouldn't based on prompt, but safety first)
        clean_lines = []
        for l in lines:
            clean = re.sub(r'^Line\d+:\s*', '', l)
            clean = clean.replace('.', '')  # Remove periods
            clean_lines.append(clean)
            
        if len(clean_lines) == len(text_list):
            return clean_lines
        else:
            print(f"[Gemini Translator] Count mismatch (In: {len(text_list)}, Out: {len(clean_lines)}). Padding...")
            # Pad or Trim
            while len(clean_lines) < len(text_list): clean_lines.append("")
            return clean_lines[:len(text_list)]
            
    return [""] * len(text_list)

def upload_gemini_file(path, api_key, mime_type="audio/mp3"):
    """Uploads file to Gemini File API and returns file_uri"""
    stats = os.stat(path)
    num_bytes = stats.st_size
    
    # 1. Start Upload
    headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(num_bytes),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json"
    }
    init_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}"
    resp = requests.post(init_url, headers=headers, json={"file": {"display_name": "audio_upload"}})
    upload_url = resp.headers.get("X-Goog-Upload-URL")
    
    # 2. Upload Content
    with open(path, 'rb') as f:
        headers_upload = {
            "Content-Length": str(num_bytes),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize"
        }
        resp = requests.post(upload_url, headers=headers_upload, data=f)
    
    if resp.status_code == 200:
        file_info = resp.json()
        return file_info['file']['uri'], file_info['file']['name']
    raise Exception(f"Upload failed: {resp.text}")

def background_process(job_id, input_path, max_chars, min_chars, is_video, ref_text=None, api_key=None, model_name=None, translate=False, split_rhythm=False):
    try:
        print(f"[{job_id}] START: Path={input_path}, KeyProvided={bool(api_key)}, Translate={translate}")
        jobs[job_id]['message'] = "Starting..."
        final_subs = []
        
        # --- GEMINI CLOUD MODE ONLY ---
        if not api_key:
            raise Exception("No API Key provided! Please check settings.")
            
        jobs[job_id]['message'] = "กำลังอัปโหลดไฟล์ไปที่ Gemini..."
        
        try:
            # Map mime type roughly
            ext = os.path.splitext(input_path)[1].lower()
            mime_type = "image/jpeg" # Should not happen
            if ext in ['.mp3']: mime_type = "audio/mp3"
            elif ext in ['.wav']: mime_type = "audio/wav"
            elif ext in ['.m4a', '.mp4', '.aac']: mime_type = "audio/mp4" 
            elif ext in ['.ogg']: mime_type = "audio/ogg"
            else: mime_type = "video/mp4"
            
            print(f"[{job_id}] Uploading as {mime_type}...")
            uri, file_name = upload_gemini_file(input_path, api_key, mime_type)
            print(f"[{job_id}] Upload Success: {uri}")
            
            # Check State
            check_url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={api_key}"
            while True:
                s_resp = requests.get(check_url).json()
                state = s_resp.get('state')
                if state == "ACTIVE": break
                if state == "FAILED": raise Exception("Gemini processing failed")
                time.sleep(2)
            
            jobs[job_id]['message'] = "Gemini กำลังถอดความ (ขอเวลาสักครู่)..."
            
            # Generate
            gen_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            prompt = """ถอดเสียงไฟล์นี้เป็นภาษาไทย:
1. แกะทุกคำพูดให้ถูกต้องที่สุด
2. [สำคัญ] เขียนภาษาไทยติดกัน "ห้ามเว้นวรรคระหว่างคำ"
3. ยกเว้นภาษาอังกฤษหรือตัวเลขให้เว้นวรรคตามปกติ
4. ส่งคำตอบกลับมาในรูปแบบ SRT Format (มี Timecode)
5. [สำคัญมาก] การแบ่งบรรทัด (Segmentation):
   - ให้ตัดบรรทัดใหม่ "ทันทีที่มีการเว้นวรรคการพูด" (Speech Pause) แม้จะเป็นวรรคสั้นๆ
   - ห้ามรวมประโยคยาวๆ เข้าด้วยกัน ให้เน้นความกระชับและตรงตามจังหวะพูด
   - ถ้าผู้พูดหยุดหายใจ ให้ขึ้นบรรทัดใหม่ทันที"""
            
            if ref_text: prompt += f"\n(คำศัพท์อ้างอิง: {ref_text[:500]})"

            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"file_data": {"mime_type": mime_type, "file_uri": uri}}
                    ]
                }],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}
            }
            
            resp = requests.post(gen_url, json=payload, timeout=300)
            if resp.status_code != 200: raise Exception(f"Gemini Gen Error: {resp.text}")
            
            raw_srt = resp.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            print(f"[{job_id}] Raw Output:\n{raw_srt[:500]}...")
            
            # Parse SRT from Gemini - support multiple timestamp formats
            clean_srt = raw_srt.replace('```srt', '').replace('```', '')
            # More flexible pattern: matches various timestamp formats
            pattern = re.compile(r'(\d{1,2}:\d{2}[:\.,]\d{2,3}(?:[,.:]\d{1,3})?)\s*-->\s*(\d{1,2}:\d{2}[:\.,]\d{2,3}(?:[,.:]\d{1,3})?)\s*\n(.*?)(?=\n\d+\n|\n\n|\Z)', re.DOTALL)
            matches = pattern.findall(clean_srt)
            
            print(f"[{job_id}] Found {len(matches)} segments.")
            
            for start_str, end_str, text_content in matches:
                start = parse_time_str(start_str)
                end = parse_time_str(end_str)
                
                clean_text = text_content.strip().replace('\n', ' ')
                # Regex to remove spaces between Thai Characters
                clean_text = re.sub(r'([ก-๙])\s+([ก-๙])', r'\1\2', clean_text)
                clean_text = re.sub(r'([ก-๙])\s+([ก-๙])', r'\1\2', clean_text)
                
                if split_rhythm:
                    # Use "Relaxed" splitting (Soft Limit ~60 chars)
                    # This allows natural pauses but prevents extremely long lines by splitting at conjunctions
                    relaxed_splits = split_text_logic(clean_text, start, end, max_chars=60, min_chars=10)
                    final_subs.extend(relaxed_splits)
                else:
                    # Original logic: Split by chars
                    splits = split_text_logic(clean_text, start, end, max_chars, min_chars)
                    final_subs.extend(splits)
                
            # --- TRANSLATION STEP ---
            if translate and final_subs:
                jobs[job_id]['message'] = "กำลังแปลภาษาอังกฤษ (Translation)..."
                print(f"[{job_id}] Starting Translation for {len(final_subs)} lines...")
                
                # Batch translate in chunks of 20
                all_thai = [s['text'] for s in final_subs]
                all_eng = []
                
                batch_size = 20
                for i in range(0, len(all_thai), batch_size):
                    batch = all_thai[i:i+batch_size]
                    translated = translate_batch_with_gemini(batch, ref_text, api_key)
                    all_eng.extend(translated)
                    
                    # Update progress message
                    jobs[job_id]['message'] = f"กำลังแปลภาษา... ({len(all_eng)}/{len(all_thai)})"
                    time.sleep(1) # Rate limit safety
                
                # Merge back
                for i, sub in enumerate(final_subs):
                    if i < len(all_eng):
                        sub['text_en'] = all_eng[i]
                    else:
                        sub['text_en'] = ""
            
            if not final_subs:
                print(f"[{job_id}] Warning: No subtitles parsed from Gemini output!")
                
        except Exception as e:
            raise Exception(f"Gemini Audio Mode Failed: {e}")

        jobs[job_id]['subtitles'] = final_subs
        jobs[job_id]['done'] = True
        jobs[job_id]['message'] = f"ถอดความเสร็จสิ้น! ({len(final_subs)} ประโยค)"
        print(f"[{job_id}] DONE. Subs={len(final_subs)}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"[{job_id}] BACKGROUND EXCEPTION: {e}")
        traceback.print_exc()
        
        # Friendly error messages
        if '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg or 'quota' in error_msg.lower():
            jobs[job_id]['error'] = "❌ API Quota หมด! กรุณารอสักครู่หรือใช้ API Key อื่น"
        elif 'API key' in error_msg or 'api_key' in error_msg.lower():
            jobs[job_id]['error'] = "❌ API Key ไม่ถูกต้อง กรุณาตรวจสอบอีกครั้ง"
        else:
            jobs[job_id]['error'] = f"❌ เกิดข้อผิดพลาด: {error_msg[:200]}"
        
        jobs[job_id]['done'] = True

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get('file')
    if not file: return "No file", 400
    
    max_chars = int(request.form.get('max_chars', 25))
    min_chars = int(request.form.get('min_chars', 15))
    job_id = str(uuid.uuid4())
    tmp_path = os.path.join(tempfile.gettempdir(), job_id + "_" + file.filename)
    file.save(tmp_path)
    
    ref_text = request.form.get('ref_text', '').strip()
    if not ref_text:
        ref_file = request.files.get('ref_file')
        if ref_file and ref_file.filename:
            try:
                content = ref_file.read().decode('utf-8', errors='ignore')
                if ref_file.filename.lower().endswith('.srt'):
                     ref_text = extract_text_from_srt(content)
                else:
                     ref_text = content
            except:
                pass

    allowed_exts = ['.mp4', '.mov', '.mkv', '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm', '.aac']
    is_video = any(file.filename.lower().endswith(ext) for ext in allowed_exts)
    
    mode_select = 'gemini' # Forced
    api_key_input = request.form.get('api_key', '').strip()
    model_select = 'gemini-2.5-flash'  # Fixed model
    do_translate = request.form.get('translate') == 'true'
    
    do_split_rhythm = request.form.get('split_mode') == 'rhythm'
    
    jobs[job_id] = {'status': 'queued', 'message': 'Queued...', 'done': False, 'filename': file.filename, 'subtitles': [], 'progress': 0}
    
    print(f"[{job_id}] Queued: File={file.filename}, Mode={mode_select}, Translate={do_translate}, Rhythm={do_split_rhythm}")
    
    threading.Thread(target=background_process, args=(job_id, tmp_path, max_chars, min_chars, is_video, ref_text, api_key_input, model_select, do_translate, do_split_rhythm)).start()
    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({"error": "Not found"}), 404
    return jsonify({
        "done": job['done'], 
        "message": job.get("message", ""), 
        "error": job.get("error"), 
        "progress": job.get("progress", 0),
        "live_preview": job.get("live_preview", "")
    })

@app.route("/get_subtitles/<job_id>")
def get_subtitles(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({"error": "Not found"}), 404
    return jsonify(job['subtitles'])

@app.route("/export_srt", methods=["POST"])
def export_srt():
    data = request.json
    filename = data.get("filename", "output.srt")
    subs = data.get("subtitles", [])
    max_chars = int(data.get("max_chars", 16))
    min_chars = int(data.get("min_chars", 10))
    
    lang = data.get("lang", "thai") # thai, eng, dual
    
    final_lines = []
    idx = 1
    
    for sub in subs:
        t_start = parse_time_str(sub['start'])
        t_end = parse_time_str(sub['end'])
        
        # Determine text content based on lang
        text_th = sub.get('text', '')
        text_en = sub.get('text_en', '')
        
        final_text = text_th
        if lang == 'eng':
            final_text = text_en if text_en else text_th # Fallback
        elif lang == 'dual':
            final_text = f"{text_th}\n{text_en}"
        
        # Since splitting is already done by backend into short lines, 
        # we might just want to use the lines as is, but maybe re-ensure constraints?
        # Actually, for English/Dual, re-splitting might break the line mapping.
        # Let's TRUST the table content (users might edit).
        
        # But wait, the original logic re-split assuming user edited long text.
        # For simplicity in v5.5, let's assume the table rows are strictly 1 subtitle block.
        # We write them out 1:1.
        
        if t_start is not None and t_end is not None and final_text:
             final_lines.append(str(idx))
             final_lines.append(f"{sub['start']} --> {sub['end']}")
             final_lines.append(final_text)
             final_lines.append("")
             idx += 1
            
    srt_content = "\r\n".join(final_lines)
    
    # Use 'utf-8-sig' to add BOM, compatible with Premiere Pro / Windows editors
    # newline='' ensures we keep our CRLF line endings without OS interference
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8-sig', delete=False, suffix='.srt', newline='') as f:
        f.write(srt_content)
        f_path = f.name
        
    @after_this_request
    def cleanup(response):
        try: os.remove(f_path)
        except: pass
        return response
    
    suffix = "_smart_edited.srt"
    if lang == 'eng': suffix = "_ENG.srt"
    elif lang == 'dual': suffix = "_DUAL.srt"
    
    name = os.path.splitext(filename)[0] + suffix
    return send_file(f_path, as_attachment=True, download_name=name)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <title>AI Auto-Caption</title>
    <style>
        body { font-family: 'Sarabun', sans-serif; background: #f4f4f9; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #333; }
        .input-group { margin-bottom: 20px; padding: 20px; border: 2px dashed #ddd; border-radius: 8px; text-align: center; }
        input[type="number"] { width: 60px; padding: 5px; }
        button { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; margin: 10px 0; }
        .sec-btn { background: #2196F3; }
        button:hover { opacity: 0.9; }
        textarea { width: 100%; height: 80px; margin-top: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        .ref-box { background: #fff8e1; border: 1px solid #ffe0b2; padding: 15px; border-radius: 8px; margin-top: 15px; }
        /* Table Styles */
        table { width: 100%; border-collapse: collapse; margin-top: 20px; display: none; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        td input { width: 100%; border: none; font-family: inherit; font-size: 14px; background: transparent; }
        td input:focus { outline: 2px solid #555; background: #fff; }
        td textarea { width: 100%; border: none; font-family: inherit; font-size: 14px; background: transparent; resize: none; overflow: hidden; min-height: 24px; }
        td textarea:focus { outline: 2px solid #555; background: #fff; }
        #progress-area { margin: 20px 0; display: none; text-align: center; }
        .progress-bar { width: 100%; height: 20px; background: #eee; border-radius: 10px; overflow: hidden; }
        .bar-fill { width: 0%; height: 100%; background: #4CAF50; transition: width 0.3s; }
        .status-text { margin-top: 5px; color: #666; font-size: 14px; }
        #download-section { text-align: center; display: none; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Auto-Caption</h1>
        
        <div id="upload-section">
            <div class="input-group">
                <button class="sec-btn" onclick="document.getElementById('file').click()" id="file-label" style="width:100%">1. เลือกวิดีโอ (Video)</button>
                <input type="file" id="file" accept=".srt,.mp4,.mov,.mp3,.wav,.m4a,.flac,.ogg,.webm,.mkv,.aac" style="display:none" onchange="document.getElementById('file-label').innerText = '✅ ' + this.files[0].name">
            </div>

            <div class="ref-box">
                <div id="gemini_settings" style="padding:15px; border:1px solid #ddd; background:#f9f9f9; border-radius:8px;">
                    <label>Gemini API Key:</label>
                    <input type="text" id="api_key" placeholder="AIzaSy..." value="{{ config['gemini_api_key'] }}" oninput="saveState()">
                    
                    <div style="margin-top:15px; padding:10px; background:#e3f2fd; border-radius:4px;">
                        <input type="checkbox" id="gen_english" onchange="saveState()">
                        <label for="gen_english" style="font-weight:bold; color:#1565c0;">🇺🇸 Generate English Subtitles (เพิ่มซับอังกฤษ)</label>
                    </div>
                    
                    <small style="color:#888;">Model: gemini-2.5-flash</small>
                </div>
                
                <button onclick="uploadFile()" id="uploadBtn">🚀 Start Process</button>
            </div>
            
            <div class="input-group" style="margin-top:15px; padding:10px; text-align: left;">
               <label style="font-weight:bold; display:block; margin-bottom:10px;">✂️ Subtitle Splitting Mode:</label>
               
               <!-- Option 1: Character Limit -->
               <div style="margin-bottom: 8px;">
                   <input type="radio" id="mode_chars" name="split_mode" value="chars" onchange="toggleMode()">
                   <label for="mode_chars">Split by Character Limit (ตัดตามจำนวนคำ)</label>
                   <div id="char_options" style="margin-left: 25px; margin-top: 5px; color: #555;">
                        Max: <input type="number" id="max" value="25" style="width:50px" oninput="saveState()"> 
                        Min: <input type="number" id="min" value="15" style="width:50px" oninput="saveState()">
                   </div>
               </div>

               <!-- Option 2: Speech Rhythm -->
               <div>
                   <input type="radio" id="mode_rhythm" name="split_mode" value="rhythm" checked onchange="toggleMode()">
                   <label for="mode_rhythm" style="color:#E91E63; font-weight:bold;">Split by Speech Rhythm (ตัดตามจังหวะพูด)</label>
               </div>
            </div>
        </div>

        <div id="progress-area">
            <div class="progress-bar"><div id="bar" class="bar-fill"></div></div>
            <div id="status" class="status-text">รอการทำงาน...</div>
        </div>

        <table id="subs-table">
            <thead>
                <tr>
                    <th width="20%">Time (Auto-Adjust)</th>
                    <th>Text (Thai)</th>
                    <th>Text (English)</th>
                </tr>
            </thead>
            <tbody id="subs-body"></tbody>
        </table>

        <div id="download-section">
            <select id="export_lang" style="padding: 10px; border-radius: 4px; border: 1px solid #ccc;">
                <option value="thai">🇹🇭 Download Thai Only</option>
                <option value="eng">🇺🇸 Download English Only</option>
                <option value="dual">🇹🇭/🇺🇸 Download Dual Subtitles</option>
            </select>
            <button onclick="exportSRT()" style="background: #2196F3; width: auto;">💾 Download SRT</button>
        </div>
    </div>

    <script>
        let currentJobId = null;
        let originalFilename = "video";

        function startUpload() {
            let fileInput = document.getElementById('file');
            if(!fileInput.files[0]) return alert("กรุณาเลือกไฟล์วิดีโอก่อน");
            
            originalFilename = fileInput.files[0].name;
            let formData = new FormData();
            formData.append("file", fileInput.files[0]);
            
            let refInput = document.getElementById('ref-file');
            if(refInput.files[0]) formData.append("ref_file", refInput.files[0]);
            
            formData.append("ref_text", document.getElementById('ref-text').value);
            formData.append("max_chars", document.getElementById('max').value);
            formData.append("min_chars", document.getElementById('min').value);

            document.getElementById('upload-section').style.display = 'none';
            document.getElementById('progress-area').style.display = 'block';
            
            let xhr = new XMLHttpRequest();
            xhr.upload.onprogress = (e) => {
                if(e.lengthComputable) {
                    let pct = Math.round((e.loaded/e.total)*100);
                    document.getElementById('bar').style.width = pct + "%";
                    document.getElementById('status').innerText = "Uploading... " + pct + "%";
                }
            };
            xhr.onload = () => {
                let res = JSON.parse(xhr.responseText);
                currentJobId = res.job_id;
                pollStatus();
            };
            xhr.open("POST", "/upload");
            xhr.send(formData);
        }

        async function uploadFile() {
            const fileInput = document.getElementById('file');
            if (fileInput.files.length === 0) {
                alert("Please select a file!");
                return;
            }

            const formData = new FormData();
            formData.append("file", fileInput.files[0]);
            formData.append("max_chars", document.getElementById('max').value);
            formData.append("min_chars", document.getElementById('min').value);
            formData.append("ref_text", "");
            
            // Add Mode Params
            formData.append("api_key", document.getElementById('api_key').value);
            formData.append("translate", document.getElementById('gen_english').checked);
            // Split Mode
            let mode = document.querySelector('input[name="split_mode"]:checked').value;
            formData.append("split_mode", mode);

            document.getElementById('uploadBtn').disabled = true;
            document.getElementById('progress-area').style.display = 'block';
            
            try {
                const response = await fetch("/upload", {
                    method: "POST",
                    body: formData
                });
                const data = await response.json();
                if (data.job_id) {
                    currentJobId = data.job_id; // Set currentJobId here
                    pollStatus();
                } else {
                    alert("Error: " + data.error);
                    document.getElementById('uploadBtn').disabled = false;
                }
            } catch (e) {
                alert("Upload failed: " + e);
                document.getElementById('uploadBtn').disabled = false;
            }
        }

        function pollStatus() {
            // Poll immediately first, then every 10 seconds
            checkStatus();
            let iv = setInterval(checkStatus, 10000);
            
            function checkStatus() {
                fetch("/status/" + currentJobId).then(r=>r.json()).then(data => {
                    if(data.done) {
                        clearInterval(iv);
                        
                        // Check for error
                        if(data.error) {
                            document.getElementById('status').innerText = data.error;
                            document.getElementById('bar').style.width = "100%";
                            document.getElementById('bar').style.backgroundColor = "#f44336";
                            document.getElementById('uploadBtn').disabled = false;
                            return;
                        }
                        
                        document.getElementById('status').innerText = "เสร็จแล้ว! กำลังโหลด...";
                        document.getElementById('bar').style.width = "100%";
                        document.getElementById('bar').style.backgroundColor = "#4CAF50";
                        loadSubtitles();
                    } else {
                        document.getElementById('status').innerText = data.message;
                        // Update progress bar based on progress value
                        let pct = data.progress || 0;
                        if(pct > 0) {
                            document.getElementById('bar').style.width = pct + "%";
                            document.getElementById('bar').style.backgroundColor = "#FF9800";
                        } else {
                            // Indeterminate state - show pulsing animation
                            document.getElementById('bar').style.width = "100%";
                            document.getElementById('bar').style.backgroundColor = "#FFE082";
                        }
                        
                        // Show live preview text
                        if(data.live_preview) {
                            let previewBox = document.getElementById('preview-box');
                            if(!previewBox) {
                                previewBox = document.createElement('div');
                                previewBox.id = 'preview-box';
                                previewBox.style.marginTop = "10px";
                                previewBox.style.padding = "10px";
                                previewBox.style.background = "#e3f2fd";
                                previewBox.style.borderRadius = "4px";
                                previewBox.style.color = "#0d47a1";
                                previewBox.style.fontStyle = "italic";
                                previewBox.style.fontSize = "14px";
                                document.getElementById('progress-area').appendChild(previewBox);
                            }
                            previewBox.innerText = '🎙️ "' + data.live_preview + '"';
                        }
                    }
                });
            }
        }

        function loadSubtitles() {
            fetch("/get_subtitles/" + currentJobId).then(r=>r.json()).then(data => {
                if(data.error) {
                    alert("Error loading subtitles: " + data.error);
                    document.getElementById('uploadBtn').disabled = false;
                    document.getElementById('progress-area').style.display = 'none';
                    document.getElementById('upload-section').style.display = 'block';
                    return;
                }

                document.getElementById('progress-area').style.display = 'none';
                
                // Show Success Message
                let existingMsg = document.getElementById('success-msg');
                if(!existingMsg) {
                    let container = document.querySelector('.container');
                    let msg = document.createElement('div');
                    msg.id = 'success-msg';
                    msg.style.padding = "15px";
                    msg.style.backgroundColor = "#e8f5e9";
                    msg.style.color = "#2e7d32";
                    msg.style.border = "1px solid #c8e6c9";
                    msg.style.borderRadius = "8px";
                    msg.style.marginBottom = "20px";
                    msg.style.textAlign = "center";
                    msg.innerHTML = "<b>✅ เรียบร้อยครับ!</b><br>เช็คข้อความในตารางด้านล่างนี้ได้เลย <br><small>แก้คำผิดให้พอใจ แล้วกดปุ่ม Save ครับ (ระบบจะคำนวณเวลาให้ใหม่เอง)</small>";
                    
                    let table = document.getElementById('subs-table');
                    table.parentNode.insertBefore(msg, table);
                }
                
                let tbody = document.getElementById("subs-body");
                tbody.innerHTML = "";
                data.forEach((sub, index) => {
                    let row = `<tr>
                        <td style="vertical-align: top;">
                            <input type="text" value="${sub.start}" onchange="updateSub(${index}, 'start', this.value)">
                            <br>
                            <input type="text" value="${sub.end}" onchange="updateSub(${index}, 'end', this.value)">
                        </td>
                        <td><textarea oninput="autoResize(this)" onchange="updateSub(${index}, 'text', this.value)" rows="1">${sub.text}</textarea></td>
                        <td><textarea oninput="autoResize(this)" onchange="updateSub(${index}, 'text_en', this.value)" rows="1" placeholder="(English)">${sub.text_en || ''}</textarea></td>
                    </tr>`;
                    tbody.innerHTML += row;
                });

                // Auto-resize all textareas after loading
                setTimeout(() => {
                    document.querySelectorAll('textarea').forEach(el => autoResize(el));
                }, 100);
                
                // Cache global subs for saving
                window.currentSubs = data;
                
                document.getElementById('subs-table').style.display = 'table';
                document.getElementById('download-section').style.display = 'block';
            });
        }

        function updateSub(index, field, value) {
            if (window.currentSubs && window.currentSubs[index]) {
                window.currentSubs[index][field] = value;
            }
        }

        function exportSRT() {
            if(!window.currentSubs) return;
            const lang = document.getElementById('export_lang').value;
            
            fetch('/export_srt', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    filename: originalFilename,
                    subtitles: window.currentSubs,
                    max_chars: document.getElementById('max').value,
                    min_chars: document.getElementById('min').value,
                    lang: lang
                })
            })
            .then(res => res.blob())
            .then(blob => {
                let url = window.URL.createObjectURL(blob);
                let a = document.createElement('a');
                a.href = url;
                a.download = originalFilename.replace(/\.[^/.]+$/, "") + (lang=='dual'?"_DUAL.srt":(lang=='eng'?"_ENG.srt":"_TH.srt"));
                document.body.appendChild(a);
                a.click();
                a.remove();
            });
        }
        function toggleMode() {
            let rythm = document.getElementById('mode_rhythm').checked;
            let maxInput = document.getElementById('max');
            let minInput = document.getElementById('min');
            let charOpts = document.getElementById('char_options');
            
            if(rythm) {
                maxInput.disabled = true;
                minInput.disabled = true;
                charOpts.style.opacity = "0.5";
            } else {
                maxInput.disabled = false;
                minInput.disabled = false;
                charOpts.style.opacity = "1";
            }
            saveState();
        }
        
        function autoResize(el) {
            el.style.height = 'auto';
            el.style.height = el.scrollHeight + 'px';
        }
        
        // Initialize UI State
        toggleMode();
    </script>
</body>
</html>
"""

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5001")

if __name__ == "__main__":
    if threading.current_thread() is threading.main_thread():
        threading.Timer(1, open_browser).start()
    app.run(port=5001, threaded=True)
