# import fitz  # PyMuPDF
# from transformers import BartForConditionalGeneration, BartTokenizer
# import textwrap 
# import streamlit as st

# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page_num in range(doc.page_count):
#         page = doc[page_num]
#         text += page.get_text()
#     doc.close()
#     return text

# def text_summarizer_from_pdf(pdf_text):
#     model_name = "facebook/bart-large-cnn"
#     model = BartForConditionalGeneration.from_pretrained(model_name)
#     tokenizer = BartTokenizer.from_pretrained(model_name)

#     inputs = tokenizer.encode("summarize: " + pdf_text, return_tensors="pt", max_length=1024, truncation=True, )
#     summary_ids = model.generate(inputs, max_length=500, min_length=180, length_penalty=2.0, num_beams=4, early_stopping=True)

#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     # formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
#     # return formatted_summary
#     return summary

# # def save_summary_as_pdf(pdf_path, summary):
# #     # Create a PDF document
# #     doc = fitz.open()

# #     # Add a new page with formatted summary
# #     page = doc.new_page()
# #     page.insert_text((20, 50), summary, fontname="helv", fontsize=15)

# #     # Save the document with the summary
# #     output_pdf_path = pdf_path.replace(".pdf", "_summary.pdf")
# #     doc.save(output_pdf_path)
# #     doc.close()

# #     return output_pdf_path


# # pdf_file_path = "firesafty.pdf"
# # summary = text_summarizer_from_pdf(pdf_file_path)
# # output_pdf_path = save_summary_as_pdf(pdf_file_path, summary)
# # print("Summary saved as PDF:", output_pdf_path)

# def main():
#     st.header("PDF")

#     with st.sidebar:
#         st.title("File Upload")
#         pdf_docs = st.file_uploader(
#             "Upload your PDF Files and Click on the Submit & Process Button", 
#             accept_multiple_files=False, 
#             type=["pdf"]
#         )

#         # result = extract_text_from_pdf(pdf_docs)
#         summary = text_summarizer_from_pdf(extract_text_from_pdf(pdf_docs))
#         print(summary)

#     if summary:
#         st.success(summary)


# if __name__ == "__main__":
#     main()