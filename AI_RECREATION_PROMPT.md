# AI Auto-Caption: AI Recreation Prompt

**Purpose**: Use this prompt to instruct an AI (like Claude, ChatGPT, Gemini) to rebuild the "AI Auto-Caption" application from scratch, ensuring all features and logic are preserved.

---

## **Prompt for AI**

**Role**: You are an expert Python developer specializing in Flask, Multimedia processing (FFmpeg), and NLP.

**Task**: Create a **Desktop-based Web Application (Flask)** called "AI Auto-Caption" that generates Thai/English subtitles from video/audio files using **Google Gemini 2.5 Flash API**. The app must be packaged as a single executable for Mac/Windows.

### **1. Core Features**
1.  **File Upload**: Support `.mp4, .mov, .mp3, .wav, .m4a`.
2.  **Transcription**: Use **Gemini 2.5 Flash API** to transcribe audio to Thai text.
3.  **Translation (Optional)**: If checked, translate Thai subtitles to English (Dual Subtitles).
4.  **Smart Splitting (NLP)**:
    - Use `pythainlp` to split Thai text correctly.
    - **Logic**: No line longer than `Max Chars` (default 25).
    - **Sticky Particles**: Words like `ครับ, ค่ะ, นะ` must stay attached to the previous sentence.
    - **Dangling Connectors**: Words like `ที่, ซึ่ง, และ` must not end a line (move to next line).
5.  **Review Interface**:
    - Show a table of generated subtitles (Start, End, Thai, English).
    - Allow users to edit text and time.
6.  **Export**:
    - Export as `.srt` file.
    - Options: Thai Only, English Only, Dual Language (Thai top / English bottom).
7.  **Progress Tracking**: Real-time progress bar (Upload -> Transcribe -> Split -> Translate).
8.  **Desktop App**: Must include build scripts (`pyinstaller`) for Mac (`.app`) and Windows (`.exe`).

### **2. Tech Stack**
-   **Backend**: Python, Flask
-   **Frontend**: HTML, CSS (Vanilla), JavaScript (Fetch API) - **Single File** (embed HTML in Python).
-   **AI**: Google Gemini API (`google-generativeai` or REST API).
-   **NLP**: `pythainlp` (for Thai words tokenization).
-   **Deployment**: `pyinstaller` for creating standalone apps.
-   **Dependencies**: `flask`, `requests`, `pythainlp`, `pandas`, `pyinstaller`.

### **3. Application Logic & Flow**

#### **A. Upload & Processing**
1.  User uploads file -> Save to temp.
2.  Upload file to Gemini File API (`upload_file`).
3.  Wait for file state to be `ACTIVE`.
4.  Send prompt to Gemini:
    > "Transcribe this audio to Thai. Return ONLY SRT format. Strict timestamps."
    > *Handle Gemini's weird timestamp formats (e.g. `MM:SS:mmm` vs `HH:MM:SS,mmm`).*
5.  Parse SRT output.

#### **B. Text Splitting (The "Secret Sauce")**
1.  Merge raw SRT segments into a large text block (sentences).
2.  Tokenize using `pythainlp.tokenize.word_tokenize(text, engine='newmm')`.
3.  Re-group tokens into subtitles based on `Max Chars` limit.
4.  **Critical Rules**:
    -   If a line ends with a "Dangling Connector" (e.g. `และ`), move it to the next line.
    -   If a line starts with a "Sticky Particle" (e.g. `ครับ`), move it to the previous line (if space permits).
5.  Re-calculate timestamps based on character count proportion.

#### **C. Translation (English)**
1.  If "Generate English" is checked:
    -   Send batch of Thai lines to Gemini.
    -   Prompt: *"Translate these Thai subtitle lines to English. Keep specific names. Return 1 line per input line."*
    -   Remove any trailing dots (`.`) from English output.
2.  Map English lines back to the corresponding Thai timestamps.

#### **D. Output**
-   Generate SRT file content.
-   For Dual mode:
    ```srt
    1
    00:00:01,000 --> 00:00:04,000
    สวัสดีครับ
    Hello

    2
    ...
    ```

### **4. UI Requirements (Single `webapp.py`)**
-   **Clean & Modern**: Use a simple card layout, green/blue accent colors.
-   **Inputs**:
    -   File Select Button.
    -   API Key Input (Stored in `config.json` or LocalStorage).
    -   Max Chars / Min Chars inputs.
    -   Checkbox: "Generate English Subtitles".
-   **Output**: Editable Table (Time | Thai | English).

---

### **Instructions for the AI Developer**
1.  **Generate `webapp.py`**: The complete Flask app with embedded HTML.
2.  **Generate `config.json`**: Template for settings.
3.  **Generate `requirements.txt`**.
4.  **Generate `build_mac.py` & `build_windows.py`**: PyInstaller scripts to bundle the app (hide console on Mac, show console on Windows for debug).
5.  **Ensure Error Handling**:
    -   Handle "Quota Exceeded" (429) gracefully.
    -   Handle Invalid API Key.

---
**Copy this entire prompt and send it to an AI to recreate the project.**
