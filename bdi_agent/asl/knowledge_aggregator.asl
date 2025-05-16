// Initial beliefs
results_dir("results").
max_papers(10).

// Goals
!start.
!create_results_directory.

// Plan to create results directory at startup
+!create_results_directory : results_dir(Dir) <-
    .print("Ensuring results directory exists");
    // Note the dot prefix
    .createDirectory(Dir).

// Plans
+!start <- .print("KnowledgeAggregatorAgent started and waiting for relevant papers").

// Plan to handle relevant papers
+relevant_papers(Data)[source(Sender)] : true <-
    .print("Received relevant papers for aggregation");
    // Note the dot prefix
    .processRelevantPapers(Data, Result);
    // Save the result to file - note the dot prefix
    .saveResults(Result);
    // Indicate completion by adding a belief
    +aggregation_completed(Result).

// Note: Recovery plans are ignored in current AgentSpeak implementation