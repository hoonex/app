import os
import sys
import streamlit as st

# 기존 RAG 관련 라이브러리 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# -----------------------------------------------------------
# 1. API 키 설정 (Secrets 사용)
# -----------------------------------------------------------
try:
    # Streamlit Secrets에서 API 키를 안전하게 가져옵니다.
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("오류: Google API Key가 Streamlit Secrets에 'GOOGLE_API_KEY'라는 이름으로 설정되지 않았습니다.")
    # API 키가 없으면 앱을 계속 진행할 수 없습니다.
    st.stop()

# 파일 설정: 파일 이름은 'rulebook.pdf'로 가정 (1단계에서 변경 요청됨)
file_path = "rulebook.pdf" 

# -----------------------------------------------------------
# 2. RAG 구성 함수 (단 한 번만 실행되도록 캐싱)
# -----------------------------------------------------------
@st.cache_resource
def setup_rag_chain():
    st.write("📖 규정집을 읽는 중...")
    # 1. PDF 로드
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except Exception as e:
        # 파일 경로 오류 메시지를 명확하게 표시하고, None을 반환하여 로딩 실패를 알림
        st.error(f"오류: PDF 파일을 불러올 수 없습니다. GitHub 저장소의 루트에 '{file_path}' 파일이 있는지 확인하세요. 에러: {e}")
        return None

    st.write("🧠 AI가 규정집을 학습 중 (임베딩 생성)...")
    # 2. 텍스트 분할 (최적화된 설정 유지)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    # 3. 벡터 DB 생성
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # 4. 리트리버 설정 (최적화된 k=15 유지)
    retriever = vector_db.as_retriever(search_kwargs={"k": 15})
    
    # 5. LLM 및 프롬프트 설정
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    prompt = ChatPromptTemplate.from_template("""
        당신은 정동고등학교 생활규정 해석 AI입니다.
        반드시 규정 조항(제몇조 몇항)을 근거로 답변하세요.
        
        규정 문장:
        {context}
        
        질문:
        {question}
        
        철저히 규정 문서에 근거해서만 답변하세요.
    """)
    
    # 6. 체인 구성
    chain = (
        RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
    )
    st.success("🎉 규정집 학습 완료!")
    return chain

# -----------------------------------------------------------
# 3. Streamlit 앱 실행 영역
# -----------------------------------------------------------

# 제목 설정
st.title("🏫 정동고등학교 학생생활규정 AI 도우미")
st.subheader("규정집을 학습한 AI에게 질문해 보세요.")

# RAG 체인 로드 (처음 실행 시 시간이 걸릴 수 있음)
rag_chain = None
with st.spinner("시스템 초기화 중입니다..."):
    rag_chain = setup_rag_chain()

# rag_chain이 성공적으로 로드되었을 경우에만 앱의 나머지 부분 실행
if rag_chain:
    # 사용자 입력 처리
    user_query = st.text_input("질문을 입력하세요 (예: 대회 입상 시 상점은 몇 점인가요?)")

    if user_query:
        # 답변 생성 및 출력
        with st.spinner("답변 생성 중..."):
            try:
                answer = rag_chain.invoke(user_query)
                st.markdown("---")
                st.markdown(f"**🤖 답변:**")
                st.info(answer.content)
            except Exception as e:
                # LLM 호출 중 발생한 오류 처리
                st.error(f"답변 생성 중 오류가 발생했습니다. API 키 또는 LLM 연결 상태를 확인하세요. 오류: {e}")
