from langchain_core.prompts import PromptTemplate

#template
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation style: "{style_input}"
Explanation Length: "{length_input}"
1. Mathematical details:
    - Include relevant mathematical equaitons if present in the paper.
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
    - Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with:"Insufficient Information" instead of guessing.
Ensure summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables=['paper_input', 'style_input', 'length_input']
)

template.save('template.json')