import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, AutoTokenizer
import torch
import PyPDF2
import os

def get_health_tip():
    return "A healthy diet and regular exercise are important for maintaining good health."

# Load the GPT-2 model and tokenizer with proper padding token setup
@st.cache_resource  # Cache the model loading
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Set the padding token to the EOS token since GPT-2 doesn't have a native padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    # Make sure the model knows about the padding token
    model.config.pad_token_id = model.config.eos_token_id
    
    return tokenizer, model

# App title
st.set_page_config(page_title="GPT-2 Chatbot")

# Function for generating GPT-2 response
def generate_gpt2_response(prompt_input, tokenizer, model):
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt_input, return_tensors="pt", padding=True)
        output = model.generate(
            input_ids, 
            max_length=512, 
            num_return_sequences=1, 
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones(input_ids.shape)  # Add attention mask
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Load data from multiple PDFs
def load_data():
    pdf_file_paths = [
        os.path.join("data", "ebooks_academic_geop4e_frontmatter.pdf"),
        os.path.join("data", "Gale-Encyclopedia-of-Psychology-2nd-ed.-2001.pdf")
    ]

    text_data = ""
    for pdf_file_path in pdf_file_paths:
        try:
            with open(pdf_file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text_data += page.extract_text()
        except FileNotFoundError:
            st.error(f"PDF file not found: {pdf_file_path}")
        except Exception as e:
            st.error(f"Error processing PDF {pdf_file_path}: {str(e)}")

    return text_data

# Initialize the model and tokenizer
try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Load the data before using it in the pipeline
try:
    data = load_data()
except Exception as e:
    st.error(f"Error loading PDF data: {str(e)}")
    data = ""

# Initialize a question-answering pipeline with proper tokenizer
qa_pipeline = pipeline(
    "question-answering", 
    model="distilbert-base-cased-distilled-squad",
    tokenizer=AutoTokenizer.from_pretrained(
        "distilbert-base-cased-distilled-squad",
        model_max_length=512
    )
)

# Store GPT-2 generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear chat history button
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', key="clear_chat_history_button", on_click=clear_chat_history)

# Process user input
def process_input(user_input):
    with st.spinner("Thinking..."):
        if "health tip" in user_input.lower():
            response = get_health_tip()
        else:
            try:
                if len(data.strip()) > 0:
                    answer = qa_pipeline(question=user_input, context=data)
                    response = answer["answer"]
                else:
                    response = generate_gpt2_response(user_input, tokenizer, model)
            except Exception as e:
                response = f"I apologize, but I encountered an error: {str(e)}"
    return response

# User prompt and response
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = process_input(prompt)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)