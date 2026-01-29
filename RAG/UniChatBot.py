from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.text_splitter import CharacterTextSplitter
import os 

# first define the model 
llm= HuggingFaceEndpoint(
    repo_id='MiniMaxAI/MiniMax-M2.1',
    task='text-generation',
)
model= ChatHuggingFace(llm=llm)
# now load the document
loader= TextLoader('RAG/Uni_information.txt',encoding='utf-8')
docs= loader.load()
# for doc in docs:
#     doc.metadata = {
#         "student_name": "Umar Ali",
#         "program": "BSCS",
#         "semester": "5"
#     }
print('document load successfully')
# now split the text into chunks 
text_spliter=CharacterTextSplitter(
            separator="\n\n",
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
        )
# now create the splits 
splits= text_spliter.split_documents(docs)
print(f'split into {len(splits)} chunks ')
# now create embeddings for documents 
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# now creating vector store
vector_store= Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    persist_directory="./chroma_db_student",
    collection_name='unidata'
)
# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
    }
)

# create prompt template 
prompt_template = """You are an intelligent  academic  agent designed to answer questions about students academic information and sheduled class information ,instrucuor information ,university information.

INSTRUCTIONS:
-Use provided information with clear and concise 
- Use  the information provided in the Context.
- Do NOT use prior knowledge or make assumptions.
- If the answer is not explicitly present in the Context, respond with:
  "I don't know based on the provided information."

RESPONSE GUIDELINES:
- Be clear, concise, and factual.
- Do not add explanations, opinions, or extra details.
- If multiple students are mentioned, ensure the answer matches the correct student.
- If the question is ambiguous, state that the information is unclear from the context.
- If someoene ask about any suggestion so you can give from your side .
- If someone greet in start so make sure to response with  greets.
-Always response like a knowledge able agent 

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
)
parser= StrOutputParser()

print("🔗 Creating QA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)
print("✅ QA chain created successfully!")
print("\n" + "="*50)
print("🎓 Student RAG System Ready!")
print("="*50)

def ask_question(question, show_sources=False):
    """Ask a question to the RAG system"""
    # print(f"\n❓ Question: {question}")
    
    try:
        result = qa_chain.invoke({"query": question})
        
        print(f"\n📝 Answer: {result['result']}")
        
        if show_sources and 'source_documents' in result:
            print(f"\n📚 Sources used: {len(result['source_documents'])}")
            for i, doc in enumerate(result['source_documents'], 1):
                content = doc.page_content
                print(f"\n--- Source {i} ---")
                # Show first 200 characters
                preview = content[:200] + "..." if len(content) > 200 else content
                print(preview)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

# Test with some questions
def test_system():
    """Test the system with sample questions"""
    test_questions = [
        "What is the student's name?",
        "What is the student's university?",
        "What subjects is the student taking?",
        "What is the student's schedule on Monday?",
        "What is the attendance percentage?",
        "Who teaches Operating System?",
    ]
    
    print("\n🧪 Testing the system...")
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {question}")
        ask_question(question, show_sources=False)
    
    print("\n✅ Testing complete!")

# Interactive mode
def interactive_mode():
    """Run interactive query mode"""
    print("\n" + "="*50)
    print("💬 Interactive Mode")
    print("="*50)
    print("\nAsk questions about the student's academic information.")
    print("Commands:")
    print("  - Type 'test' to run sample questions")
    print("  - Type 'sources on/off' to toggle source display")
    print("  - Type 'quit', 'exit', or 'q' to end")
    print("="*50)
    
    show_sources = True
    
    while True:
        try:
            user_input = input("\n💭 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
                
            elif user_input.lower() == 'test':
                test_system()
                continue
                
            elif user_input.lower() == 'sources on':
                show_sources = True
                print("✅ Source display enabled")
                continue
                
            elif user_input.lower() == 'sources off':
                show_sources = False
                print("✅ Source display disabled")
                continue
                
            elif not user_input:
                continue
                
            # Process the question
            ask_question(user_input, show_sources)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

# Main execution
if __name__ == "__main__":
    import sys
    
    # Check if question provided as argument
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ask_question(question)
    else:
        # Run interactive mode
        interactive_mode()



