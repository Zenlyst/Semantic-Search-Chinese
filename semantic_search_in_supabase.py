import re
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from supabase import create_client, Client

def main():
    # initialize supabase client
    url = "https://your_supabase_url"
    key = "your_supabase_key"
    supabase: Client = create_client(url, key)

    ########### load data from supabase ##########
    embeddings_model = OpenAIEmbeddings()
    response = supabase.table("tablename").select("question, answer, created_at, id, video_url").execute()
    data = response.data 
    created_at = []
    question = []
    ids = []
    answer = []
    video_url = []

    for item in data:
        ids.append(item['id'])
        created_at.append(item['created_at'])
        question.append(item['question'])
        answer.append(item['answer'])
        video_url.append(item['video_url'])

    ######### generate embedding ###########
    embedding = embeddings_model.embed_documents(question)

    ######### Write embedding to the supabase table  #######
    # Consider using update instead of insert to avoid duplicates
    for id, new_embedding in zip(ids, embedding):
        supabase.table("tablename").update({"embedding": new_embedding.tolist()}).eq("id", id).execute()

    ########### Vector Store #############
    # Put pre-compute embeddings to vector store. ## save to disk
    vectorstore = Chroma.from_texts(
        texts=question,
        embedding=embeddings_model,
        persist_directory="./chroma_db_carbon_questions"
    )

    ###### load from disk  #######
    vectorstore = Chroma(persist_directory="./chroma_db_carbon_questions", embedding_function=embeddings_model)

    ####### Query it #########
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
    target = ["台北101有什麼好玩的？"]
    query = [
        "台北101的觀景台怎麼去",
        "台北101附近有什麼美食",
        "台北101的門票多少錢",
        "台北101有什麼特色活動",
        "去台北101要注意什麼"
    ]
    evaluation(target, query)

if __name__ == "__main__":
    main()