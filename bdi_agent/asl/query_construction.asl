// Initial beliefs
api_key("default_key").

// Initial goal
!start.

// Plans
+!start <- .print("Query Construction Agent started").

+research_query(Question)[source(Sender)] <-
    .print("Received research question: ", Question);
    .constructQuery(Question, Params);
    .print("Created search parameters: ", Params);
    +query_result(Params).
        