# **⚠️ NOTE: This is a prototype version meant for testing structure and functionality. A more advanced implementation using a fine-tuned transformer (e.g., T5 or better) with ethical filters and detailed training will be released soon.**

# Mental Health Chatbot (GPT-2 Based)

## Overview
The Mental Health Chatbot is an AI-powered conversational agent designed to assist users with mental health support. Built using a GPT-2 model, it provides responses to user queries while maintaining a sensitive and supportive tone. The chatbot operates on a Python backend with a Streamlit UI, offering an accessible and interactive experience for users.

## Features
- **Natural Language Processing**: Uses a GPT-2 model to generate responses.
- **User-Friendly UI**: Built using Streamlit for a smooth and interactive experience.
- **Mental Health Focus**: Provides responses related to mental well-being.
- **Easy Deployment**: Can be hosted locally or on a cloud platform.

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- Virtual environment (optional but recommended)

## Installation

1. **Clone the Repository**
```bash
git clone https://github.com/DHariharanD/AlienAlgos-Chatbot.git
cd mental-health-chatbot
```

2. **Create and Activate a Virtual Environment (Optional)**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## Usage

1. **Run the Chatbot**
```bash
streamlit run chatbotgpt2.py
```

2. **Interact with the Chatbot**
   - Open your browser and navigate to the displayed Streamlit URL.
   - Enter queries related to mental health and receive AI-generated responses.

## Configuration
- You can modify the chatbot's parameters in `config.py`.
- Adjust the response model parameters to fine-tune chatbot behavior.

## Deployment
To deploy on a cloud platform (e.g., Heroku, AWS, or Streamlit Sharing), follow the respective service’s instructions.

## Contributing
Feel free to contribute by submitting issues, feature requests, or pull requests.

## Contact
For any queries, contact [harisharan200425@gmail.com] or raise an issue in the repository.
