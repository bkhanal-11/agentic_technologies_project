- A system of specialized agents that collaborate to conduct comprehensive literature review.
- The application is in Python and is implemented using SPADE
- Here's the documentation of SPADE: https://spade-mas.readthedocs.io/en/latest/ Before answering any questions, consult the documentation.
- The agents we will implement are:
    - Knowledge agent (src/agents/knowledge.py): It will fetch and store the papers and their content using the HTML version (either using ar5ive or using arxiv.org/html for newer papers) and Jina Reader API (curl "https://r.jina.ai/URL" \
  -H "Authorization: Bearer API_KEY")
    - Analysis agent (src/agents/analysis.py): Reads each paper's content and extract the methodology, findings and future work and save them.
    - Summarization agent (src/agents/summarization.py): Reads the outputs of the analysis agent and summarizes them as a whole.
    - Synthesis agent (src/agents/synthesis.py): Reads the outputs of the summarization agent and constructs a report on the findings, common themes, and research gaps with recommendations for the researcher in that field.



- The current knowledge base structure is a json file under `./src/knowledge_bases`:
```json
{
  "research_question": "",
  "papers": [
    {
      "id": "",
      "title": "",
      "abstract": "",
      "authors": [],
      "relevance_score": 0.0,
      "url": ""
    },
    ...
  ],
  "timestamp": ""
}
```