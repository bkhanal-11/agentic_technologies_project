
// Initial beliefs
max_papers(10).
content_priority(fulltext).  // Could be: abstract, fulltext, hybrid
ready(true).

// Initial goals
!setup.

// Plan for setup
+!setup
   <- .print("KnowledgeAggregatorAgent initialized with BDI architecture").

// Plan for handling new relevant papers
+new_relevant_papers(Question, Papers)
   <- .print("Received relevant papers for: ", Question);
      +current_question(Question);
      +papers_to_process(Papers);
      !create_knowledge_base(Question, Papers).

// Plan for creating knowledge base
+!create_knowledge_base(Question, Papers)
   <- .print("Creating knowledge base for: ", Question);
      .create_knowledge_folder(Question, FolderPath);
      +knowledge_folder(FolderPath);
      !process_paper_content(Papers).

// Plan for processing paper content
+!process_paper_content(Papers)
   <- .print("Processing paper content");
      ?knowledge_folder(FolderPath);
      ?content_priority(Priority);
      ?max_papers(MaxPapers);
      .process_papers(FolderPath, Papers, Priority, MaxPapers, ProcessedCount);
      +papers_processed(ProcessedCount);
      !save_research_data.

// Plan for saving research data
+!save_research_data
   <- .print("Saving research data");
      ?knowledge_folder(FolderPath);
      ?current_question(Question);
      ?papers_to_process(Papers);
      .save_research_json(FolderPath, Question, Papers);
      !notify_analysis_agent.

// Plan for notifying analysis agent
+!notify_analysis_agent
   <- .print("Notifying analysis agent");
      ?knowledge_folder(FolderPath);
      ?current_question(Question);
      ?papers_processed(Count);
      .print("Processed ", Count, " papers");
      .notify_analysis_agent(FolderPath, Question);
      -new_relevant_papers(Question, _);
      -current_question(_);
      -papers_to_process(_);
      -knowledge_folder(_);
      -papers_processed(_).
