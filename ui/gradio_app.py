import gradio as gr

from app.agent.graph import research_graph


def research(question: str) -> str:
    if not question.strip():
        return "Please enter a research question."
    result = research_graph.invoke({"user_query": question, "paper_ids": []})
    return result["answer"]


with gr.Blocks(title="Scientific Research Assistant") as demo:
    gr.Markdown(
        "# Scientific Research Assistant\nLocal arXiv search, lazy ingestion, and cited answers."
    )
    question = gr.Textbox(label="Research question", lines=3)
    run = gr.Button("Research", variant="primary")
    answer = gr.Markdown()
    run.click(research, inputs=question, outputs=answer)


if __name__ == "__main__":
    demo.launch()
