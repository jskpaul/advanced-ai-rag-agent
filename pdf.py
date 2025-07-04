import os
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.readers.file import PDFReader

def get_index(data, index_name):
    
    index = None
    if not os.path.exists(index_name):
        print("building index...", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
    
    return index


pdf_path = os.path.join("data", "Canada.pdf")
canada_pdf = PDFReader().load_data(file = pdf_path)

canada_index = get_index(canada_pdf, "canada")
canada_engine = canada_index.as_query_engine()

south_korea_pdf = PDFReader().load_data(file=os.path.join("data", "South_Korea.pdf"))
south_korea_index = get_index(south_korea_pdf, "south_korea")
south_korea_engine = south_korea_index.as_query_engine()
