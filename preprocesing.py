# preprocess_textbooks.py
import PyPDF2
import os
import json
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from tqdm import tqdm

def extract_text_from_pdfs():
    pdf_file_paths = [
        os.path.join("data", "ebooks_academic_geop4e_frontmatter.pdf"),
        os.path.join("data", "Gale-Encyclopedia-of-Psychology-2nd-ed.-2001.pdf")
    ]
    
    text_chunks = []
    
    for pdf_path in tqdm(pdf_file_paths, desc="Processing PDFs"):
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # Process each page
                for page_num in range(len(pdf_reader.pages)):
                    text = pdf_reader.pages[page_num].extract_text()
                    
                    # Split text into smaller chunks (e.g., paragraphs)
                    paragraphs = text.split('\n\n')
                    for para in paragraphs:
                        if len(para.strip()) > 50:  # Only keep substantial paragraphs
                            text_chunks.append({
                                'content': para.strip(),
                                'source': os.path.basename(pdf_path),
                                'page': page_num + 1
                            })
                            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
    
    return text_chunks

def create_embeddings(text_chunks):
    # Initialize the model and tokenizer for creating embeddings
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    embeddings = []
    
    # Process chunks in batches
    for chunk in tqdm(text_chunks, desc="Creating embeddings"):
        # Tokenize and get embeddings
        inputs = tokenizer(chunk['content'], padding=True, truncation=True, 
                         max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling to get sentence embedding
            embedding = outputs.last_hidden_state.mean(dim=1)
            
        embeddings.append({
            'content': chunk['content'],
            'source': chunk['source'],
            'page': chunk['page'],
            'embedding': embedding[0].tolist()  # Convert tensor to list for JSON storage
        })
    
    return embeddings

def main():
    # Create data directory if it doesn't exist
    os.makedirs('processed_data', exist_ok=True)
    
    # Extract text from PDFs
    print("Extracting text from PDFs...")
    text_chunks = extract_text_from_pdfs()
    
    # Create embeddings
    print("Creating embeddings...")
    knowledge_base = create_embeddings(text_chunks)
    
    # Save processed data
    print("Saving processed data...")
    with open('processed_data/knowledge_base.json', 'w') as f:
        json.dump(knowledge_base, f)
    
    print("Preprocessing complete! Knowledge base saved to processed_data/knowledge_base.json")

if __name__ == "__main__":
    main()