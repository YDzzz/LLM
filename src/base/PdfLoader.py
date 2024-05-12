from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:
    @staticmethod
    def loader(file_path: str):
        content = ""
        loader = PyPDFLoader(file_path)
        for page in loader.load():
            content += page.page_content
        return content


if __name__ == '__main__':
    print(PDFLoader.loader("C:\\Users\\16922\\Desktop\\文档1.pdf"))
