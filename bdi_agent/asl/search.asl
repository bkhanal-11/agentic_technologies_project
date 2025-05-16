// Initial beliefs
max_results(10).
timeout(60).

// Goals
!start.

// Plans
+!start <- .print("SearchAgent started and waiting for search parameters").

// Plan to handle search_params belief addition
+search_params(Params)[source(Sender)] : true <-
    .print("Received search parameters");
    // Parse parameters from JSON string - note the dot prefix
    .parseParams(Params, ResearchQuestion);
    // Execute search using custom action - note the dot prefix
    .executeSearch(Params, Results);
    // Send results to the RelevantAgent
    .send(relevant_agent, tell, search_results(Results, ResearchQuestion)).

// Note: Recovery plans are ignored in current AgentSpeak implementation