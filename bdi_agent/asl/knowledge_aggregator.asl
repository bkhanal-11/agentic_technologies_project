// Initial beliefs
ready(true).

// Initial goals
!setup.

// Plans
+!setup
   <- .print("KnowledgeAggregatorAgent initialized with BDI architecture").

// Plan for handling relevant papers
+relevant_papers(Question, Papers)
   <- .print("Received relevant papers for question: ", Question);
      .register_relevant_papers(Question, Papers);
      -relevant_papers(Question, Papers).