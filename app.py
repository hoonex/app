import os
import sys
import streamlit as st # Streamlit 라이브러리 임포트

# 기존 RAG 관련 라이브러리 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# -----------------------------------------------------------
# 1. API 키 설정
# -----------------------------------------------------------
# Streamlit Secrets에서 API 키를 가져옵니다.
import streamlit as st
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("오류: Google API Key가 Streamlit Secrets에 설정되지 않았습니다.")

# 파일 설정: 파일 이름을 단순화 (rulebook.pdf)
# 사용자님께서 반드시 PDF 파일 이름을 'rulebook.pdf'로 변경하고 GitHub에 올려야 합니다.
file_path = "rulebook.pdf" 

# -----------------------------------------------------------
# 2. RAG 구성 함수 (단 한 번만 실행되도록 캐싱)
# -----------------------------------------------------------
@st.cache_resource
def setup_rag_chain():
    # 1. PDF 로드
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except Exception as e:
        # 파일 경로 오류 메시지를 더 명확하게 변경
        st.error(f"오류: PDF 파일을 불러올 수 없습니다. GitHub 저장소의 루트에 '{file_path}' 파일이 있는지 확인하세요. 에러: {e}")
        sys.exit()

    # 2. 텍스트 분할 (최적화된 설정 유지)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    # 3. 벡터 DB 생성
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # 4. 리트리버 설정
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
    return chain

# -----------------------------------------------------------
# 3. Streamlit 앱 실행 영역
# -----------------------------------------------------------

# 제목 설정
st.title("🏫 정동고등학교 학생생활규정 AI 도우미")
st.subheader("규정집을 학습한 AI에게 질문해 보세요.")

# RAG 체인 로드 (처음 실행 시 시간이 걸릴 수 있음)
try:
    with st.spinner("규정집을 학습 중입니다... 잠시만 기다려 주세요."):
        rag_chain = setup_rag_chain()

    # 사용자 입력 처리
    user_query = st.text_input("질문을 입력하세요 (예: 대회 입상 시 상점은 몇 점인가요?)")

    if user_query:
        # 답변 생성 및 출력
        with st.spinner("답변 생성 중..."):
            answer = rag_chain.invoke(user_query)
            st.markdown("---")
            st.markdown(f"**🤖 답변:**")
            st.info(answer.content)
            
except SystemExit:
    # PDF 로드 실패 시 앱 종료를 처리 (st.error 메시지는 이미 출력됨)
    pass
except Exception as e:
    # 기타 오류 처리
    st.error(f"앱 실행 중 알 수 없는 오류가 발생했습니다: {e}")

        except Exception as e:

            st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
