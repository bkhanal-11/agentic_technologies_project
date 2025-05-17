// Initial beliefs
// None

// Plans
// Plan triggered when analysis is ready
+analysis_ready(Data)[source(Sender)] : true
    <- .print("PLAN FIRED: Received notification that analysis is ready");
       .synthesize_report(Data).

// Plan when synthesis is complete
+synthesis_complete("true") : true
    <- .print("PLAN FIRED: Synthesis complete, literature review process finished").

// Plan to handle errors
+synthesis_error(Error) : true
    <- .print("PLAN FIRED: Error in synthesis: ", Error).