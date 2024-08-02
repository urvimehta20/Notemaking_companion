# Python Notemaking Companion
A tool designed to help you with your Python note-taking journey. This application leverages the power of a retrieval-augmented generation (RAG) pipeline to provide helpful and accurate answers to your Python-related questions. 

The RAG pipeline is built on an e-book named **"Python Notes for Professionals"** which is around 850 pages of content.

## Demo
https://github.com/user-attachments/assets/7cfe09a3-1420-44d9-8047-69ddbd91e539

## Features:
- Interactive PDF Display
- Context Aware Responses
- Source Document Highlighting
- Caching for Optimal Performance
- Open-source LLM Integration

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/urvimehta20/Notemaking_companion.git
    cd https://github.com/urvimehta20/Notemaking_companion.git
    ```

2. **Set up a virtual environment**:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    Create an `apikey.env` file in the project directory with the following content:
    ```sh
    HUGGINGFACEHUB_API_TOKEN="your_huggingface_api_token"
    ```

## Usage

1. **Run the Streamlit application**:
    ```sh
    streamlit run rag.py
    ```

2. **Interact with the Application**:
    - Enter your Python-related question in the text area.
    - Click the "Ask" button to get a response.
    - The response will be displayed in the main area, and the relevant PDF will be shown in the sidebar.
