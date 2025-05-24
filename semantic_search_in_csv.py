import re
from langchain_community.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

############## Load data from csv ###############

def extract_question(text):
    match = re.search(r'Question: (.+?)(?:\n|$)', text)
    if match:
        return match.group(1)
    return None

def main():
    # Load CSV data
    loader = CSVLoader(file_path="./QA_database.csv")  # Fixed path: add '.' if file is in current dir
    data = loader.load()

    # Extract questions
    questions = [extract_question(doc.page_content) for doc in data if extract_question(doc.page_content)]

    # Generate embeddings
    embeddings_model = OpenAIEmbeddings()
    embedding = embeddings_model.embed_documents(questions)

    # Vector Store: save to disk
    vectorstore = Chroma.from_texts(
        texts=questions,
        embedding=embeddings_model,
        persist_directory="./chroma_db_carbon_questions"
    )

    # Load from disk
    vectorstore = Chroma(persist_directory="./chroma_db_carbon_questions", embedding_function=embeddings_model)

    # Evaluation function
    def evaluation(target, query, SIMILARITY_THRESHOLD=0.83):
        correct = 0
        count = 0   
        print(f"問題：{target[0]}")
        for q in query:
            docs_and_scores = vectorstore.similarity_search_with_relevance_scores(q, k=1)
            doc, score = docs_and_scores[0]
            print(f"Variation{count+1}:{q}")
            if doc.page_content in target: 
                print(f"找到Target | score:{round(score, 2)}")
                correct += 1 
            elif score >= SIMILARITY_THRESHOLD: 
                print(f"找到近似問題：{doc.page_content} | score:{round(score, 2)}")
            else:
                print(f"沒有找到 ｜ score:{round(score, 2)}")
            count += 1
        print(f"Accuracy rate {correct} / {count}")

    # Example usage
    target = ["台灣為什麼要制定氣候變遷因應法？"]
    query = [
        "氣候變遷因應法是什麼時候制定的",
        "氣候變遷因應法有什麼內容",
        "氣候變遷有哪些法律",
        "台灣有哪些氣候法規",
        "氣候變遷的影響"
    ]
    evaluation(target, query)

if __name__ == "__main__":
    main()