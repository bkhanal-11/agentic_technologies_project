// Initial beliefs
// None

// Plans
// Plan triggered when receiving a research query
+research_query(Question)[source(Sender)] : true
    <- .print("PLAN FIRED: Received research query: ", Question);
       +research_question(Question);
       .generate_search_params(Question, "false").

// Plan triggered when receiving a refined query request
+refined_query(Question, PrevResults)[source(Sender)] : true
    <- .print("PLAN FIRED: Received refined query: ", Question);
       +research_question(Question);
       +previous_results(PrevResults);
       +is_refined("true");
       .generate_search_params(Question, "true").

// Plan to send search parameters when the params_ready belief is added
+params_ready(X) : true
    <- .print("PLAN FIRED: Search parameters ready, triggering send goal");
       !send_search_params.

// Achievement goal plan for sending search parameters
+!send_search_params : true
    <- .print("PLAN FIRED: Executing send_search_params goal");
       .send_search_params.

// Plan to handle errors
+params_error(Error) : true
    <- .print("PLAN FIRED: Error generating search parameters: ", Error).