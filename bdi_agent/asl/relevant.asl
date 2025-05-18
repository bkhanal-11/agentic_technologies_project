// Initial beliefs
relevance_threshold(0.7).
ready(true).

// Initial goals
!setup.

// Plans
+!setup
   <- .print("RelevantAgent initialized with BDI architecture").

// Plan for handling search results
+search_results(Question, Results)
   <- .print("Received search results for question: ", Question);
      .register_search_results(Question, Results);
      -search_results(Question, Results).