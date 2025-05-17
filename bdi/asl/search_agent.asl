// Initial beliefs
// None

// Plans
// Plan triggered when receiving search parameters
+search_params(Params)[source(Sender)] : true
    <- .print("PLAN FIRED: Received search parameters");
       .perform_search(Params).

// Plan to trigger the goal when results are ready
+results_ready(X) : true
    <- .print("PLAN FIRED: Search results ready with value: ", X);
       !send_search_results.

// Achievement goal plan for sending search results
+!send_search_results : true
    <- .print("PLAN FIRED: Executing send_search_results goal");
       .send_search_results.

// Plan to handle errors
+search_error(Error) : true
    <- .print("PLAN FIRED: Error in search: ", Error).