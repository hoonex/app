import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

file_path = "rulebook.pdf" 
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVER_K = 50
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("오류: Google API Key가 Streamlit Secrets에 'GOOGLE_API_KEY'라는 이름으로 설정되지 않았습니다.")
    st.stop()

@st.cache_resource
def setup_rag_chain():
    st.write("규정집을 읽는 중...")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except Exception as e:
        st.error(f"오류: PDF 파일을 불러올 수 없습니다. '{file_path}' 파일이 GitHub 루트에 있는지 확인하세요. 에러: {e}")
        return None 

    st.write("AI가 규정집을 학습 중...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.1)
    
    prompt = ChatPromptTemplate.from_template("""
    당신은 정동고등학교 생활규정 해석 AI입니다.
    1. 반드시 조항(제몇조 몇항)과 벌점/상점 점수('숫자')를 포함하여 답변하세요. 
    2.사용자가 한 질문들을 매우 넓게 포괄적으로 생각하세요.
    3. 불필요하거나 일반적인 조항(예: 총칙, 목적, 징계의 일반 원칙 등)은 대부분 제외하고, 오직 제112조나 113조의 구체적인 항목 중 관련성 높은 항목만 최대 3개를 제시하세요. 
    규정 문장: {context} 
    질문: {question} 
    철저히 검색된 규정 문서에 근거해서만 답변하세요.
    특히 벌점 및 상점은 검색된 문서 내용의 '표, 리스트, 목록'을 참고하여 정확한 점수를 찾아 비교하여 제시하세요.
    """)
    
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | llm
    )
    st.success("학습 완료!")
    return chain

# 3. Streamlit 앱 실행
st.title("🏫 정동고 학생생활규정 도우미")
st.subheader("규정집을 학습한 도우미에게 질문해 보세요.")

rag_chain = None
with st.spinner("시스템 초기화 중입니다..."):
    rag_chain = setup_rag_chain()

if rag_chain:
    user_query = st.text_input("질문을 입력하세요.")

    if user_query:
        with st.spinner("답변 생성 중..."):
            try:
                answer = rag_chain.invoke(user_query)
                st.markdown("---")
                st.markdown(f"답변:")
                st.info(answer.content)
            except Exception as e:
                st.error(f"답변 생성 중 오류가 발생했습니다. 오류: {e}")









