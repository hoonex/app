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
가장 중요한 규칙:
규정집 내에서 질문과 1%라도 관련된 조항, 벌점, 상점이 존재하면 반드시 모두 찾아내고 절대 누락하지 마십시오.  
불확실하거나 애매해도 '관련 가능성이 조금이라도 있으면' 무조건 제시하십시오.  
규정집에서 숫자를 찾을 수 있으면 반드시 숫자를 제시하십시오.  
숫자가 명확하지 않을 때만 "모르겠습니다"라고 표시합니다. 추측은 금지합니다.
        규정 문장: {context}
        질문: {question}
출력 형식 규칙(강제):
- 최대 3개의 항목을 제시하되, 규정집에서 숫자가 있는 항목을 최우선적으로 선택하십시오.
- 각 항목은 아래 3줄만 포함:
  1) 조항: 제{조}조 {몇항}
  2) 규정문구: 문서의 해당 문장을 15~30자 이내로 요약 또는 인용
  3) 벌점/상점 숫자: 규정집의 표·리스트·벌점표·상점표에 있는 정확한 숫자만 기입
     (문서에 숫자가 있으면 무조건 적기, 조금이라도 관련 있으면 무조건 적기)

검색 및 추출 기준:
- 질문 내용과 직접적 또는 간접적 관련이 있는 조항을 모두 탐색하십시오.
- 관련될 가능성이 있다면 반드시 선택하여 숫자를 제시하십시오.
- 문서 내에서 숫자가 존재하면 절대 생략 금지.
- 규정의 한 항목이 여러 표와 연결될 경우, 숫자가 있는 표를 우선 적용하여 제시.

예외 처리:
- 규정집에 전혀 숫자가 없는 경우에만 "모르겠습니다"로 표시.
- 규정집에 조항은 있지만 숫자가 없으면 조항은 적고 숫자만 "모르겠습니다"라고 표시.

이 규칙들은 절대적으로 우선 적용되며, 응답 형식을 변경하지 마십시오.
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







