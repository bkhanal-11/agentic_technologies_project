// Initial beliefs
// None

// Plans
// Plan triggered when receiving search results
+search_results(Results)[source(Sender)] : true
    <- .print("PLAN FIRED: Received search results");
       .evaluate_relevance(Results).

// Plan to refine query if needed and insufficient relevant papers
+evaluation_complete(X) : should_refine("false") | (num_relevant_papers(N) & N >= 5)
    <- .print("PLAN FIRED: Need to refine query due to insufficient relevant papers");
       !request_query_refinement.

// Goal to request query refinement
+!request_query_refinement : true
    <- .print("PLAN FIRED: Executing request_query_refinement goal");
       .request_query_refinement.

// Plan to send relevant papers to knowledge aggregator
+evaluation_complete(X) : should_refine("false") | num_relevant_papers(N) & N >= "5"
    <- .print("PLAN FIRED: Found sufficient relevant papers, sending to knowledge aggregator");
       !send_relevant_papers.

// Goal to send relevant papers
+!send_relevant_papers : true
    <- .print("PLAN FIRED: Executing send_relevant_papers goal");
       .send_relevant_papers.

// Plan to handle errors
+evaluation_error(Error) : true
    <- .print("PLAN FIRED: Error evaluating relevance: ", Error).