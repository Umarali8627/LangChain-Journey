from langchain_core.prompts import PromptTemplate



template=PromptTemplate(
    template="""please summarize the research paper titled "{paper_input}" with the followeing specifications:
    Explaination Style: {summary_style}
    Explanation Length: {summary_length}
    Provide a concise summary highlighting the key points and findings of the paper.

    if certain details are not available in the paper, respond with "Details not available in the paper."
    """,
    input_variables=["paper_input", "summary_style", "summary_length"],
    validate_template=True
)

template.save("template.json")