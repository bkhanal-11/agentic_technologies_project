// Initial beliefs
// None

// Plans
// Plan triggered when knowledge is ready
+knowledge_ready(Data)[source(Sender)] : true
    <- .print("PLAN FIRED: Received notification that knowledge is ready");
       .analyze_papers(Data).

// Plan to notify synthesis agent when analysis is complete
+analysis_complete("true") : true
    <- .print("PLAN FIRED: Analysis complete, notifying synthesis agent");
       .notify_synthesis_agent.

// Plan to handle errors
+analysis_error(Error) : true
    <- .print("PLAN FIRED: Error in analysis: ", Error).