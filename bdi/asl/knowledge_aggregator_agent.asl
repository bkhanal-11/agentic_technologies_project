// Initial beliefs
// None

// Plans
// Plan triggered when receiving relevant papers
+relevant_papers(Data)[source(Sender)] : true
    <- .print("PLAN FIRED: Received relevant papers data");
       .aggregate_knowledge(Data).

// Plan to notify analysis agent when aggregation is complete
+aggregation_complete("true") : true
    <- .print("PLAN FIRED: Knowledge aggregation complete, notifying analysis agent");
       .notify_analysis_agent.

// Plan to handle errors
+aggregation_error(Error) : true
    <- .print("PLAN FIRED: Error in knowledge aggregation: ", Error).