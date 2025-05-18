
// Initial beliefs
ready(true).
quality_threshold(0.7).
search_strategy(comprehensive).  // Could be: basic, comprehensive, focused

// Initial goals
!setup.

// Plan for setup
+!setup
   <- .print("QueryConstructionAgent initialized with BDI architecture").

// Plan for when a new query is received
+new_query(Question)
   <- .print("Received research question: ", Question);
      +current_query(Question);
      !analyze_query(Question);
      !generate_search_params(Question).

// Plan for analyzing a query
+!analyze_query(Question)
   <- .print("Analyzing query complexity and domain...");
      ?search_strategy(Strategy);
      if (Strategy == comprehensive) {
          +query_domain(scientific);
          +num_queries_needed(3);
      } else {
          +query_domain(general);
          +num_queries_needed(1);
      }.

// Plan for generating search parameters
+!generate_search_params(Question)
   <- .print("Generating search parameters for query: ", Question);
      ?query_domain(Domain);
      ?num_queries_needed(NumQueries);
      .create_search_queries(Question, Domain, NumQueries, Queries);
      +search_queries(Queries);
      !submit_search_request(Question, Queries).

// Plan for submitting search request
+!submit_search_request(Question, Queries)
   <- .print("Submitting search request to SearchAgent");
      .send_search_params("search_agent@localhost", Question, Queries);
      -new_query(Question);
      -current_query(Question);
      +query_processing_complete(Question).

// Plan for handling refined query requests
+refined_query(Question, PreviousResults)
   <- .print("Received refined query request for: ", Question);
      +current_query(Question);
      +refine_context(PreviousResults);
      !create_refined_search_params(Question).

// Plan for creating refined search parameters
+!create_refined_search_params(Question)
   <- .print("Creating refined search parameters");
      ?refine_context(PreviousResults);
      .create_refined_queries(Question, PreviousResults, RefinedQueries);
      +refined_search_queries(RefinedQueries);
      !submit_refined_search_request(Question, RefinedQueries).

// Plan for submitting refined search request
+!submit_refined_search_request(Question, Queries)
   <- .print("Submitting refined search request");
      .send_search_params("search_agent@localhost", Question, Queries);
      -refined_query(Question, _);
      -current_query(Question);
      -refine_context(_);
      +refine_complete(Question).
