from langchain_core.prompts import PromptTemplate

paper_prompt = PromptTemplate(
    input_variables = ["context"],

    template = """
    You are an expert AI Researcher.
    
    Analyze the research paper content below.
    
    Provide : 
    
    1. Executive Summary
    2. Problem Statement
    3. Key Contributions
    4. Methodology
    5. Datasets used
    6. Results
    7. Strengths
    8. Weaknesses
    9. Future work
    10. Explain like I'm 20
    
    Paper content : 
    
    {context}
    """
)

