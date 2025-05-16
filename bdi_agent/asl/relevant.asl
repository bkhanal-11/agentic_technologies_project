// Initial beliefs
relevance_threshold(0.7).
min_papers(5).

// Goals
!start.

// Plans
+!start <- .print("RelevantAgent started and waiting for search results").

// Plan to handle search results
+search_results(Results, ResearchQuestion)[source(Sender)] : true <-
    .print("Received search results for processing");
    // Note the dot prefix for all actions
    .evaluateRelevance(Results, ResearchQuestion, EvaluationResult);
    // Check if we need to refine the query
    .needsRefinement(EvaluationResult, ShouldRefine, RelevantPapers, RefinementSuggestion);
    
    // Use if-else construct in AgentSpeak
    if (ShouldRefine = "true") {
        .print("Query refinement needed, sending request to query construction agent");
        // Create refinement request - note the dot prefix
        .prepareRefinementRequest(ResearchQuestion, RefinementSuggestion, RelevantPapers, Request);
        // Send to query construction agent
        .send(query_construction_agent, tell, refine_query(Request));
    } else {
        .print("Found sufficient relevant papers, sending to knowledge aggregator");
        // Prepare data for knowledge aggregator - note the dot prefix
        .preparePapersForAggregation(ResearchQuestion, RelevantPapers, AggregationData);
        // Send to knowledge aggregator agent
        .send(knowledge_aggregator_agent, tell, relevant_papers(AggregationData));
    }.

// Plan to handle evaluation failure - uses dot-prefixed createFallbackData
-!search_results(Results, ResearchQuestion)[error(Error), error_msg(Msg)] <-
    .print("Error during relevance evaluation: ", Msg);
    // Create fallback data using a Python action instead of string concatenation
    .createFallbackData(ResearchQuestion, FallbackData);
    .send(knowledge_aggregator_agent, tell, relevant_papers(FallbackData)).