// Initial beliefs
ready(true).

// Initial goals
!setup.

// Plans
+!setup
   <- .print("QueryConstructionAgent initialized with BDI architecture").

// Plan for handling a new research query
+research_query(Question)
   <- .print("Received research question: ", Question);
      !process_query(Question).

// Simple plan to process regular queries
+!process_query(Question)
   <- .print("Processing query: ", Question);
      .process_regular_query(Question);
      -research_query(Question).