
// Initial beliefs
relevance_threshold(0.7).
refinement_threshold(5).  // Minimum papers needed to avoid refinement
ready(true).

// Initial goals
!setup.

// Plan for setup
+!setup
   <- .print("RelevantAgent initialized with BDI architecture").

// Plan for handling new search results
+new_search_results(Question, Results)
   <- .print("Received search results for question: ", Question);
      +current_question(Question);
      +raw_results(Results);
      !evaluate_paper_relevance(Question, Results).

// Plan for evaluating paper relevance
+!evaluate_paper_relevance(Question, Results)
   <- .print("Evaluating relevance of papers");
      ?relevance_threshold(Threshold);
      .evaluate_relevance(Question, Results, Threshold, EvaluatedResults, RelevanceData);
      +evaluated_results(EvaluatedResults);
      +relevance_data(RelevanceData);
      !decide_next_action(Question).

// Plan for deciding what to do next
+!decide_next_action(Question)
   <- ?relevance_data(RelevanceData);
      ?evaluated_results(EvaluatedResults);
      ?refinement_threshold(MinPapers);
      .count_relevant_papers(EvaluatedResults, RelevantCount);
      .should_refine_query(RelevanceData, ShouldRefine);
      +relevant_paper_count(RelevantCount);
      +should_refine(ShouldRefine);
      .print("Found ", RelevantCount, " relevant papers. Should refine: ", ShouldRefine);
      if (ShouldRefine & RelevantCount < MinPapers) {
          !request_query_refinement(Question);
      } else {
          !forward_relevant_papers(Question);
      }.

// Plan for requesting query refinement
+!request_query_refinement(Question)
   <- .print("Requesting query refinement");
      ?evaluated_results(EvaluatedResults);
      ?relevance_data(RelevanceData);
      .extract_refinement_suggestion(RelevanceData, Suggestion);
      .extract_paper_ids(EvaluatedResults, PaperIds);
      .send_refinement_request("query_construction_agent@localhost", Question, PaperIds, Suggestion);
      -new_search_results(Question, _);
      -current_question(_);
      -raw_results(_);
      -evaluated_results(_);
      -relevance_data(_);
      -relevant_paper_count(_);
      -should_refine(_).

// Plan for forwarding relevant papers
+!forward_relevant_papers(Question)
   <- .print("Forwarding relevant papers to KnowledgeAggregator");
      ?evaluated_results(RelevantPapers);
      .send_relevant_papers_message("knowledge_aggregator_agent@localhost", Question, RelevantPapers);
      -new_search_results(Question, _);
      -current_question(_);
      -raw_results(_);
      -evaluated_results(_);
      -relevance_data(_);
      -relevant_paper_count(_);
      -should_refine(_).
