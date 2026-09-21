args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: run_sony_blflowsom_unlabeled_metaclustering.R <input_manifest.csv> <output_dir>")
manifest_path <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
output <- normalizePath(args[[2]], winslash = "/", mustWork = FALSE)
dir.create(output, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(FlowSOM))
manifest <- read.csv(manifest_path, stringsAsFactors = FALSE, check.names = FALSE)
required <- c("parent_seed", "codes_path", "nodes_path")
if (!all(required %in% colnames(manifest))) stop("input manifest missing required columns")
if (nrow(manifest) != 2L || !identical(sort(as.integer(manifest$parent_seed)), c(1L, 2L))) stop("expected parent seeds 1 and 2")

capture.output(
  cat("FlowSOM version:", as.character(packageVersion("FlowSOM")), "\n\n"),
  print(FlowSOM::MetaClustering),
  print(FlowSOM:::DetermineNumberOfClusters),
  print(FlowSOM:::findElbow),
  print(FlowSOM:::SSE),
  print(FlowSOM::metaClustering_consensus),
  file = file.path(output, "flowsom_metaclustering_function_bodies.txt")
)

synthetic <- outer(1:100, 1:5, function(i, j) sin(i / (j + 1)) + cos(i * j / 31))
qualification_k <- FlowSOM:::DetermineNumberOfClusters(synthetic, max = 12, method = "metaClustering_consensus", seed = 24680)
qualification_auto <- as.integer(FlowSOM::MetaClustering(synthetic, "metaClustering_consensus", max = 12, seed = 24680))
qualification_direct <- as.integer(FlowSOM::metaClustering_consensus(synthetic, k = qualification_k, seed = 24680))

summary_rows <- list()
mapping_rows <- list()
add_result <- function(parent_seed, regime, selector_seed, requested_k, selected_k, labels, elapsed, label_information) {
  if (length(labels) != 100L) stop("node mapping must have 100 rows")
  idx <- length(summary_rows) + 1L
  summary_rows[[idx]] <<- data.frame(
    dataset = "Levine_13dim", parent_seed = parent_seed, regime = regime,
    selector_seed = selector_seed, requested_k = requested_k, selected_k = selected_k,
    occupied_node_metaclusters = length(unique(labels)), r_runtime_seconds = elapsed,
    label_information_used_for_regime_definition = label_information,
    labels_used_by_r_metaclustering = FALSE,
    stringsAsFactors = FALSE
  )
  mapping_rows[[idx]] <<- data.frame(
    dataset = "Levine_13dim", parent_seed = parent_seed, regime = regime,
    selector_seed = selector_seed, requested_k = requested_k, selected_k = selected_k,
    node_index_1based = 1:100, metacluster_label_1based = as.integer(labels),
    stringsAsFactors = FALSE
  )
}

for (i in seq_len(nrow(manifest))) {
  row <- manifest[i, ]
  codes <- as.matrix(read.csv(row$codes_path, header = FALSE, check.names = FALSE))
  storage.mode(codes) <- "double"
  if (!identical(dim(codes), c(100L, 13L))) stop("unexpected codes shape for parent seed ", row$parent_seed)

  started <- proc.time()[[3L]]
  auto <- as.integer(FlowSOM::MetaClustering(codes, "metaClustering_consensus", max = 40, seed = 12345))
  add_result(row$parent_seed, "official_auto_max40", 12345L, 40L, length(unique(auto)), auto, proc.time()[[3L]] - started, FALSE)

  for (fixed_k in c(10L, 40L, 24L)) {
    started <- proc.time()[[3L]]
    labels <- as.integer(FlowSOM::metaClustering_consensus(codes, k = fixed_k, seed = 12345))
    regime <- if (fixed_k == 24L) "official_fixed_ktrue_24" else paste0("official_fixed_k", fixed_k)
    add_result(row$parent_seed, regime, 12345L, fixed_k, fixed_k, labels, proc.time()[[3L]] - started, fixed_k == 24L)
  }
  cat("parent_seed=", row$parent_seed, " auto_k=", length(unique(auto)), "\n", sep = "")
  flush.console()
}

row <- manifest[manifest$parent_seed == 1L, ][1L, ]
codes <- as.matrix(read.csv(row$codes_path, header = FALSE, check.names = FALSE))
storage.mode(codes) <- "double"
for (selector_seed in 0:29) {
  started <- proc.time()[[3L]]
  labels <- as.integer(FlowSOM::MetaClustering(codes, "metaClustering_consensus", max = 40, seed = selector_seed))
  add_result(1L, "official_auto_max40_selector_seed_sensitivity", selector_seed, 40L, length(unique(labels)), labels, proc.time()[[3L]] - started, FALSE)
}

summary <- do.call(rbind, summary_rows)
mappings <- do.call(rbind, mapping_rows)
write.csv(summary, file.path(output, "r_run_summary.csv"), row.names = FALSE)
write.csv(mappings, file.path(output, "node_metacluster_mappings.csv"), row.names = FALSE)

main <- summary$regime != "official_auto_max40_selector_seed_sensitivity"
checks <- data.frame(
  check = c("flowsom_version", "qualification_k_range", "qualification_wrapper_exact", "summary_rows", "main_rows", "selector_rows", "mapping_rows", "selected_k_range", "fixed_k_contract", "label_permission_contract"),
  passed = c(
    as.character(packageVersion("FlowSOM")) == "2.18.0",
    qualification_k >= 2L && qualification_k <= 12L,
    identical(qualification_auto, qualification_direct),
    nrow(summary) == 38L,
    sum(main) == 8L,
    sum(!main) == 30L,
    nrow(mappings) == 3800L,
    all(summary$selected_k >= 2L & summary$selected_k <= 40L),
    all(summary$selected_k[grepl("official_fixed", summary$regime)] == summary$requested_k[grepl("official_fixed", summary$regime)]),
    all(!summary$labels_used_by_r_metaclustering) && all(summary$label_information_used_for_regime_definition == (summary$regime == "official_fixed_ktrue_24"))
  ),
  detail = c(
    as.character(packageVersion("FlowSOM")), qualification_k, paste("labels", length(qualification_auto)),
    nrow(summary), sum(main), sum(!main), nrow(mappings), paste(range(summary$selected_k), collapse = "-"),
    "fixed selected_k equals requested_k", "only Ktrue definition uses label count; no labels enter R"
  ), stringsAsFactors = FALSE
)
write.csv(checks, file.path(output, "r_checks.csv"), row.names = FALSE)
capture.output(sessionInfo(), file = file.path(output, "r_session_info.txt"))
cat("checks=", sum(checks$passed), "/", nrow(checks), "\n", sep = "")
if (!all(checks$passed)) quit(status = 1L)
