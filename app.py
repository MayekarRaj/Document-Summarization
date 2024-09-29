import os
import logging
from io import BytesIO
from dotenv import load_dotenv
from gtts import gTTS
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
import streamlit as st
from streamlit_markmap import markmap
# from transformers import pipeline
from transformers import BartForConditionalGeneration, BartTokenizer
import fitz  # PyMuPDF
import textwrap
import torch
import wordninja
import nltk
import re
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab') # or punkt
nltk.download('stopwords')

st.set_page_config("Document Dialogue", layout="wide")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key not found.")
else:
    genai.configure(api_key=api_key)

logging.basicConfig(level=logging.INFO)

# @st.cache_resource
# def load_summarizer():
#     try:
#         device = 0 if torch.cuda.is_available() else -1
#         summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
#         logging.info(f"Summarization model loaded on device: {device}")
#         return summarizer
#     except Exception as e:
#         logging.error(f"Error loading summarizer: {e}")
#         st.error("Failed to load the summarization model.")
#         return None

# summarizer = load_summarizer()

def gemini_image(image):

    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = 'Generate a detailed description of the image provided.'
    try:
        chain = model.generate_content(contents=[prompt, image])
        chain.resolve()
        return chain.text
    except Exception as e:
        logging.error(f"Error generating image description: {e}")
        return "Image description unavailable."

def clean_text(text):
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    
    text = re.sub(r'\.(?!\s)', r'. ', text)
    text = re.sub(r',(?=\S)', r', ', text)
    
    text = re.sub(r'-\s+', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = split_concatenated_words(text, min_word_length=15)
    
    sentences = sent_tokenize(text)
    sentences = [s.strip().capitalize() for s in sentences]
    text = ' '.join(sentences)
    
    return text.strip()

def getpdf(pdf_file):
    text = ""
    # for pdf_file in pdf_files:
    try:
        pdf_reader = PdfReader(pdf_file)
        for page_num, page in enumerate(pdf_reader.pages):
            page_content = page.extract_text()
            if page_content:
                cleaned_page = clean_text(page_content)
                text += cleaned_page + "\n"
                # text += page_content
            try:
                xobject = page.get("/Resources", {}).get("/XObject", {})
                for obj in xobject.values():
                    obj_type = obj.get("/Subtype", None)
                    if obj_type == "/Image":
                        image_data = obj.get_data()
                        image_file = BytesIO(image_data)
                        image_description = gemini_image(image_file)
                        text += f"\nPage {page_num + 1}: Image Description: {image_description}\n"
            except Exception as e:
                logging.warning(f"No images found on page {page_num + 1}: {e}")
    except Exception as e:
        logging.error(f"Error reading PDF file {pdf_file}: {e}")
        st.error(f"Failed to read PDF file {pdf_file}.")

    return text                  

def get_chunks(data, chunk_size=1000, chunk_overlap=200):
    text_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_split.split_text(data)
    logging.info(f"Text split into {len(chunks)} chunks.")
    return chunks

def get_vector(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local('vector_store')
        logging.info("Vector store created and saved locally.")
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")

def get_accuracy(ai_response, pdf_data):
    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = ('Give me an accuracy in percentage of how much the data are related to each other '
              'only return the percentage and the accuracy should be between 70-100% it cannot be anything else')
    try:
        chain = model.generate_content(contents=[prompt, str(ai_response), str(pdf_data)])
        chain.resolve()
        return chain.text
    except Exception as e:
        logging.error(f"Error in get_accuracy: {e}")
        return "Error calculating accuracy"

def get_conversation_chain(temp, top_k, top_p): 
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    The user wants to chat with the PDF, so help them out by ensuring the user is not disappointed with the response. 
    Analyze the context thoroughly as much as possible. Avoid speculation and focus on verifiable information.
    
    # Context:
    {context} 
    
    # Question: 
    {question} 
    
    # Answer: 
    """
    try:
        model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=temp, top_k=top_k, top_p=top_p)
        prompt = PromptTemplate(template=prompt_template, input_variables=['context','question'])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        logging.info("Conversation chain initialized successfully.")
        return chain
    except Exception as e:
        logging.error(f"Error initializing conversation chain: {e}")
        st.error("Failed to initialize the conversation chain.")
        return None

def split_concatenated_words(text, min_word_length=15):
    words = text.split()
    split_words = []
    for word in words:
        if len(word) > min_word_length:
            split = wordninja.split(word)
            split_words.extend(split)
        else:
            split_words.append(word)
    return ' '.join(split_words)

def user_input(user_question, temp, top_k, top_p):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        if not docs:
            st.warning("No relevant documents found.")
            return
        
        chain = get_conversation_chain(temp, top_k, top_p)
        if not chain:
            st.error("Conversation chain is not available.")
            return
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        st.success(response["output_text"])
        # Uncomment the following lines to enable text-to-speech and accuracy
        # speak_text(response['output_text'])
        # acc = get_accuracy(response["output_text"], docs[0])
        # st.write("Accuracy: ", acc)
    except Exception as e:
        logging.error(f"Error in user_input: {e}")
        st.error("An error occurred while processing your input. Please try again.")

def speak_text(text):
    try:
        os.makedirs("temp", exist_ok=True)
        tts = gTTS(text, lang="en")
        tts.save("temp/temp.mp3")
        audio_file = open("temp/temp.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/ogg")
        st.markdown(""" 
            <style>
            .stAudio{
                width: 300px !important;
            }</style>
            """, unsafe_allow_html=True)
    except Exception as e:
        logging.error(f"Error in speak_text: {e}")
        st.write("Error in text-to-speech.")

def generate_markdown(text):
    query = rf"""
        Study the given {text} and generate a summary then please be precise in selecting the data such that it gets to a hierarchical structure. 
        Don't give anything else, I just want to display the structure as a mindmap so be precise please. 
        Don't write anything else, Just return the md file. It is not necessary to cover all information. 
        Don't use triple backticks or ` anywhere. Cover the main topics. Please convert this data into a markdown mindmap format similar to the following example:
        ---
        markmap:
        colorFreezeLevel: 2
        ---
    
        # Gemini Account Summary
    
        ## Balances
    
        - Bitcoin (BTC): 0.1234
        - Ethereum (ETH): 0.5678
    
        ## Orders
    
        - Open Orders
        - Buy Order (BTC): 0.01 BTC @ $40,000
        - Trade History
        - Sold 0.1 ETH for USD at $2,500
    
        ## Resources
    
        - [Gemini Website](https://www.gemini.com/)
    """
    model = genai.GenerativeModel('gemini-1.5-pro')
    try:
        chain = model.generate_content(contents=[query])
        chain.resolve()
        markmap(chain.text)
    except Exception as e:
        logging.error(f"Error generating markdown: {e}")
        st.error("An error occurred while generating the mindmap.")


# def generate_bart_summary(text, summarizer, max_length, min_length):
#     try:
#         # Split text into manageable chunks
#         chunks = get_chunks(text, chunk_size=1000, chunk_overlap=200)
#         summaries = []
        
#         for chunk in chunks:
#             try:
#                 summary_result = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
#                 summary = summary_result[0]['summary_text']
#                 summaries.append(summary)
#             except Exception as e:
#                 st.error("An error occurred while generating the summary.")
#                 logging.error(f"Error summarizing chunk: {e}")
        
#         final_summary = ' '.join(summaries)
        
#         word_count = len(final_summary.split())
#         st.session_state.summary_word_count = word_count
        
#         logging.info("Summary generation completed successfully.")
#         return final_summary
#     except Exception as e:
#         logging.error(f"Error generating BART summary: {e}")
#         return "Summary generation unavailable."

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

def load_model_and_tokenizer():
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return model, tokenizer

def text_summarizer_from_pdf(pdf_text, max_len=500, min_len=180):
    model, tokenizer = load_model_and_tokenizer()

    inputs = tokenizer.encode("summarize: " + pdf_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_len, min_length=min_len, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = "\n".join(textwrap.wrap(summary, width=80))

    # Store word count in session state
    st.session_state.summary_word_count = len(formatted_summary.split())
    return formatted_summary

# def text_summarizer_from_pdf(pdf_text, max_len = 500, min_len = 180):
#     model_name = "facebook/bart-large-cnn"
#     model = BartForConditionalGeneration.from_pretrained(model_name)
#     tokenizer = BartTokenizer.from_pretrained(model_name)

#     inputs = tokenizer.encode("summarize: " + pdf_text, return_tensors="pt", max_length=1024, truncation=True, )
#     summary_ids = model.generate(inputs, max_length=max_len, min_length=min_len, length_penalty=2.0, num_beams=4, early_stopping=True)

#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
#     # word_count = len(formatted_summary)
#     st.session_state.summary_word_count = len(formatted_summary.split())
#     return formatted_summary
#     # return summary

def words_to_tokens(words):
    """
    Converts words to approximate token count.

    Parameters:
    - words (int): Number of words.

    Returns:
    - int: Approximate token count.
    """
    return int(words / 0.75)

def tokens_to_words(tokens):
    """
    Converts tokens to approximate word count.

    Parameters:
    - tokens (int): Number of tokens.

    Returns:
    - int: Approximate word count.
    """
    return int(tokens * 0.75)

def main():
    st.header("Chat with PDF Seamlessly")
    
    if 'raw_text' not in st.session_state:
        st.session_state.raw_text = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""
    if 'summary_word_count' not in st.session_state:
        st.session_state.summary_word_count = 0
    
    with st.sidebar:
        st.title("File Upload")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            # accept_multiple_files=True,
            accept_multiple_files=False, 
            type=["pdf"]
        )

        st.header("Summarization Settings")
        max_length_words = st.slider("Maximum Summary Length (words)", min_value=120, max_value=400, value=250)
        max_length = words_to_tokens(max_length_words)
        min_length_words = st.slider("Minimum Summary Length (words)", min_value=50, max_value=200, value=180)
        min_length = words_to_tokens(min_length_words)

        st.markdown(f"**Converted to tokens:**\n Max Length = {max_length} tokens, Min Length = {min_length} tokens")
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing... May take more than a few minutes"):
                    try:
                        raw_text = getpdf(pdf_docs)
                        if not raw_text.strip():
                            st.warning("No text extracted from the PDF.")
                        else:
                            # summary = generate_bart_summary(raw_text, summarizer, max_length=max_length, min_length=min_length)
                            summary = text_summarizer_from_pdf(extract_text_from_pdf(pdf_docs), max_length, min_length)

                            st.session_state.raw_text = raw_text
                            st.session_state.summary = summary
                            st.success("Processing completed!")
                    except Exception as e:
                        logging.error(f"Error processing PDF: {e}")
                        st.error("An error occurred while processing your PDF.")
    
        st.header("Prompt Settings")
        temperature = st.slider("Select Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        top_k = st.slider("Select Top-k", min_value=1, max_value=100, value=50, step=1)
        top_p = st.slider("Select Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.05)    
    
    if st.session_state.summary:
        st.subheader("Summary of the PDF:")
        st.write(st.session_state.summary)
        st.write(f"**Word Count:** {st.session_state.summary_word_count} words")
    
    user_question = st.text_input("Ask a Question from the PDF", key='user_question')
    if st.button("Submit Question"):
        if st.session_state.user_question.strip():
            with st.spinner("Generating answer..."):
                user_input(st.session_state.user_question, temperature, top_k, top_p)
        else:
            st.warning("Please enter a question to submit.")

    if st.button('Generate a Mindmap'):
        if st.session_state.raw_text:
            with st.spinner("Generating mindmap..."):
                try:
                    generate_markdown(st.session_state.raw_text)
                except Exception as e:
                    logging.error(f"Error generating mindmap: {e}")
                    st.error("An error occurred while generating the mindmap.")
        else:
            st.warning("Please upload and process a PDF first.")

if __name__ == "__main__":
    main()