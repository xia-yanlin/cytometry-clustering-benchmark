args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: run_flowsom_r_rare_metaclustering_batch.R <input_manifest.csv> <output_dir>")
manifest_path <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
output <- normalizePath(args[[2]], winslash = "/", mustWork = FALSE)
dir.create(output, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(FlowSOM))
manifest <- read.csv(manifest_path, stringsAsFactors = FALSE, check.names = FALSE)
required <- c("dataset", "som_seed", "codes_path")
if (!all(required %in% colnames(manifest))) stop("input manifest missing required columns")
if (nrow(manifest) != 60L) stop("expected 60 parent SOM code matrices")

capture.output(
  cat("FlowSOM version:", as.character(packageVersion("FlowSOM")), "\n\n"),
  print(FlowSOM::MetaClustering), print(FlowSOM:::DetermineNumberOfClusters),
  print(FlowSOM:::findElbow), print(FlowSOM:::SSE), print(FlowSOM::metaClustering_consensus),
  file = file.path(output, "flowsom_rare_metaclustering_function_bodies.txt")
)

# FlowSOM::metaClustering_consensus(data, k=2) asks ConsensusClusterPlus for
# maxK=2.  Its cluster-tracking plot then receives an empty matrix and fails
# after the k=2 consensus partition has already been computed.  Requesting
# maxK=3 through FlowSOM's own internal consensus helper and taking result[[2]]
# preserves the same official k=2 consensus computation while bypassing only
# that unrelated plotting boundary.
fixed_consensus <- function(data, k, seed) {
  if (k == 2L) {
    results <- FlowSOM:::consensus(data, max = 3L, seed = seed)
    return(as.integer(results[[2L]]$consensusClass))
  }
  as.integer(FlowSOM::metaClustering_consensus(data, k = k, seed = seed))
}

synthetic <- outer(1:100, 1:5, function(i, j) sin(i / (j + 1)) + cos(i * j / 31))
qualification_selected <- FlowSOM:::DetermineNumberOfClusters(synthetic, max = 12, method = "metaClustering_consensus", seed = 24680)
qualification_auto <- as.integer(FlowSOM::MetaClustering(synthetic, "metaClustering_consensus", max = 12, seed = 24680))
qualification_direct <- as.integer(FlowSOM::metaClustering_consensus(synthetic, k = qualification_selected, seed = 24680))
qualification_k2 <- fixed_consensus(synthetic, 2L, 24680)
capture.output(print(fixed_consensus), file = file.path(output, "k2_boundary_adapter_function.txt"))

mapping_rows <- list(); summary_rows <- list()
add_result <- function(dataset, som_seed, regime, selector_seed, requested_k, selected_k, labels, elapsed) {
  if (length(labels) != 100L) stop("node mapping must have 100 rows")
  idx <- length(summary_rows) + 1L
  summary_rows[[idx]] <<- data.frame(
    dataset = dataset, som_seed = som_seed, regime = regime, selector_seed = selector_seed,
    requested_k = requested_k, selected_k = selected_k,
    occupied_node_metaclusters = length(unique(labels)), r_runtime_seconds = elapsed,
    stringsAsFactors = FALSE
  )
  mapping_rows[[idx]] <<- data.frame(
    dataset = dataset, som_seed = som_seed, regime = regime, selector_seed = selector_seed,
    requested_k = requested_k, selected_k = selected_k, node_index_0based = 0:99,
    metacluster_label_1based = as.integer(labels), stringsAsFactors = FALSE
  )
}

for (i in seq_len(nrow(manifest))) {
  row <- manifest[i, ]
  codes <- as.matrix(read.csv(row$codes_path, check.names = FALSE)); storage.mode(codes) <- "double"
  if (nrow(codes) != 100L) stop("unexpected code rows")
  regimes <- list(
    official_auto_max40 = c(40L, NA_integer_), official_fixed_ktrue_2 = c(2L, 2L),
    official_fixed_k10 = c(10L, 10L), official_fixed_k40 = c(40L, 40L)
  )
  for (regime in names(regimes)) {
    started <- proc.time()[[3L]]
    if (regime == "official_auto_max40") {
      labels <- as.integer(FlowSOM::MetaClustering(codes, "metaClustering_consensus", max = 40, seed = 12345))
      selected <- length(unique(labels))
    } else {
      selected <- regimes[[regime]][[2L]]
      labels <- fixed_consensus(codes, k = selected, seed = 12345)
    }
    elapsed <- proc.time()[[3L]] - started
    add_result(row$dataset, row$som_seed, regime, 12345L, regimes[[regime]][[1L]], selected, labels, elapsed)
  }
  cat(row$dataset, " som_seed=", row$som_seed, " complete\n", sep = ""); flush.console()
}

for (dataset in unique(manifest$dataset)) {
  row <- manifest[manifest$dataset == dataset & manifest$som_seed == 0L, ][1L, ]
  codes <- as.matrix(read.csv(row$codes_path, check.names = FALSE)); storage.mode(codes) <- "double"
  for (selector_seed in 0:29) {
    started <- proc.time()[[3L]]
    labels <- as.integer(FlowSOM::MetaClustering(codes, "metaClustering_consensus", max = 40, seed = selector_seed))
    elapsed <- proc.time()[[3L]] - started
    add_result(dataset, 0L, "official_auto_max40_selector_seed_sensitivity", selector_seed, 40L, length(unique(labels)), labels, elapsed)
  }
}

summary <- do.call(rbind, summary_rows); mappings <- do.call(rbind, mapping_rows)
write.csv(summary, file.path(output, "r_run_summary.csv"), row.names = FALSE)
write.csv(mappings, file.path(output, "node_metacluster_mappings.csv"), row.names = FALSE)
checks <- data.frame(
  check = c("flowsom_version", "qualification_selected_range", "qualification_wrapper_and_k2", "main_rows", "selector_rows", "mapping_rows", "selected_ranges", "fixed_k_contract", "dataset_counts"),
  passed = c(
    as.character(packageVersion("FlowSOM")) == "2.18.0", qualification_selected >= 2L && qualification_selected <= 12L,
    identical(qualification_auto, qualification_direct) && length(qualification_k2) == 100L && length(unique(qualification_k2)) == 2L, sum(summary$regime != "official_auto_max40_selector_seed_sensitivity") == 240L,
    sum(summary$regime == "official_auto_max40_selector_seed_sensitivity") == 60L, nrow(mappings) == 30000L,
    all(summary$selected_k >= 2L & summary$selected_k <= 40L),
    all(summary$selected_k[summary$regime == "official_fixed_ktrue_2"] == 2L) && all(summary$selected_k[summary$regime == "official_fixed_k10"] == 10L) && all(summary$selected_k[summary$regime == "official_fixed_k40"] == 40L),
    all(table(summary$dataset) == 150L)
  ),
  detail = c(as.character(packageVersion("FlowSOM")), qualification_selected, paste("auto", length(qualification_auto), "k2", length(unique(qualification_k2))), 240, 60, nrow(mappings), paste(range(summary$selected_k), collapse = "-"), "2/10/40", paste(table(summary$dataset), collapse = ";")),
  stringsAsFactors = FALSE
)
write.csv(checks, file.path(output, "r_checks.csv"), row.names = FALSE)
capture.output(sessionInfo(), file = file.path(output, "r_session_info.txt"))
cat("checks=", sum(checks$passed), "/", nrow(checks), "\n", sep = "")
if (!all(checks$passed)) quit(status = 1L)
