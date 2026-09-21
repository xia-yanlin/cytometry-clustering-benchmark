args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7L) {
  stop("usage: run_flowsom_r_full_event_batch.R <source_csv> <marker_file> <cofactor> <dataset> <output_dir> <max_new_runs> <protocol_sha256>")
}

source_path <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
marker_path <- normalizePath(args[[2]], winslash = "/", mustWork = TRUE)
cofactor <- as.numeric(args[[3]])
dataset <- args[[4]]
output <- normalizePath(args[[5]], winslash = "/", mustWork = FALSE)
max_new_runs <- as.integer(args[[6]])
protocol_sha256 <- args[[7]]
dir.create(output, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(FlowSOM))
suppressPackageStartupMessages(library(jsonlite))

contracts <- list(
  Samusik_01 = list(rows = 86864L, markers = 39L),
  Levine_32dim = list(rows = 265627L, markers = 32L)
)
if (!dataset %in% names(contracts)) stop("unsupported dataset")
contract <- contracts[[dataset]]
markers <- readLines(marker_path, warn = FALSE, encoding = "UTF-8")
if (length(markers) != contract$markers || anyDuplicated(markers)) stop("marker contract failed")

header <- names(read.csv(source_path, nrows = 0L, check.names = FALSE))
if (!all(markers %in% header)) stop("marker columns missing")
column_classes <- ifelse(header %in% markers, "numeric", "NULL")
started_read <- proc.time()[[3L]]
frame <- read.csv(source_path, check.names = FALSE, colClasses = column_classes)
matrix_value <- as.matrix(frame[, markers, drop = FALSE])
storage.mode(matrix_value) <- "double"
rm(frame)
if (!identical(dim(matrix_value), c(contract$rows, contract$markers))) stop("matrix dimension contract failed")
if (!all(is.finite(matrix_value))) stop("non-finite raw marker value")
matrix_value <- asinh(matrix_value / cofactor)
if (!all(is.finite(matrix_value))) stop("non-finite transformed marker value")
read_seconds <- proc.time()[[3L]] - started_read

capture.output(
  cat("FlowSOM version:", as.character(packageVersion("FlowSOM")), "\n\n"),
  print(FlowSOM::FlowSOM), print(FlowSOM::SOM), print(FlowSOM::BuildSOM),
  print(FlowSOM::BuildMST), print(FlowSOM::metaClustering_consensus),
  file = file.path(output, "r_flowsom_function_bodies.txt")
)
capture.output(sessionInfo(), file = file.path(output, "r_session_info.txt"))
write.csv(data.frame(
  dataset = dataset, source = source_path, rows = nrow(matrix_value), markers = ncol(matrix_value),
  cofactor = cofactor, transform = "asinh(x/cofactor)", scale = FALSE,
  label_column_loaded = FALSE, read_transform_seconds = read_seconds,
  protocol_sha256 = protocol_sha256, stringsAsFactors = FALSE
), file.path(output, "r_input_contract.csv"), row.names = FALSE)

write_int32 <- function(values, path) {
  connection <- file(path, open = "wb")
  on.exit(close(connection), add = TRUE)
  writeBin(as.integer(values), connection, size = 4L, endian = "little")
}

completed_seeds <- function() {
  seeds <- integer()
  for (seed in 0:29) {
    manifest <- file.path(output, "runs", sprintf("seed%03d", seed), "r_run_manifest.json")
    if (file.exists(manifest)) seeds <- c(seeds, seed)
  }
  seeds
}

write_progress <- function() {
  seeds <- completed_seeds()
  write.csv(data.frame(
    dataset = dataset, completed_runs = length(seeds), expected_runs = 30L,
    completed_seeds = paste(seeds, collapse = ";"),
    status = if (length(seeds) == 30L) "r_complete" else "r_in_progress",
    stringsAsFactors = FALSE
  ), file.path(output, "r_progress.csv"), row.names = FALSE)
}

dir.create(file.path(output, "runs"), showWarnings = FALSE)
write_progress()
new_runs <- 0L
for (seed in 0:29) {
  run <- file.path(output, "runs", sprintf("seed%03d", seed))
  manifest_path <- file.path(run, "r_run_manifest.json")
  if (file.exists(manifest_path)) next
  if (max_new_runs > 0L && new_runs >= max_new_runs) break
  if (dir.exists(run)) stop("incomplete run preserved: ", run)
  dir.create(run, recursive = TRUE)
  tryCatch({
    started <- proc.time()[[3L]]
    fsom <- FlowSOM::FlowSOM(
      matrix_value, transform = FALSE, scale = FALSE, silent = TRUE,
      xdim = 10L, ydim = 10L, rlen = 30L, nClus = 40L, seed = seed
    )
    runtime <- proc.time()[[3L]] - started
    node_labels <- as.integer(fsom$map$mapping[, 1L])
    bmu_distance <- as.numeric(fsom$map$mapping[, 2L])
    node_metaclusters <- as.integer(fsom$metaclustering)
    event_metaclusters <- node_metaclusters[node_labels]
    codes <- as.matrix(fsom$map$codes)
    write_int32(node_labels, file.path(run, "event_node_labels.int32le"))
    write_int32(event_metaclusters, file.path(run, "event_metacluster_labels.int32le"))
    write.csv(codes, file.path(run, "som_codes.csv"), row.names = FALSE)
    write.csv(data.frame(node_index_1based = 1:100, metacluster_label_1based = node_metaclusters), file.path(run, "node_metaclusters.csv"), row.names = FALSE)
    summary <- data.frame(
      dataset = dataset, seed = seed, n_events = length(node_labels), n_markers = ncol(codes),
      xdim = 10L, ydim = 10L, rlen = 30L, requested_metaclusters = 40L,
      occupied_nodes = length(unique(node_labels)), occupied_metaclusters = length(unique(event_metaclusters)),
      mean_bmu_distance = mean(bmu_distance), median_bmu_distance = median(bmu_distance),
      runtime_seconds = runtime, labels_used_for_fit = FALSE, stringsAsFactors = FALSE
    )
    write.csv(summary, file.path(run, "r_run_summary.csv"), row.names = FALSE)
    checks <- data.frame(
      check = c("flowsom_version", "event_count", "code_shape", "node_label_range", "event_meta_count", "node_meta_count", "metacluster_range", "finite_codes", "finite_bmu", "labels_permission"),
      passed = c(
        as.character(packageVersion("FlowSOM")) == "2.18.0", length(node_labels) == contract$rows,
        identical(dim(codes), c(100L, contract$markers)), min(node_labels) >= 1L && max(node_labels) <= 100L,
        length(event_metaclusters) == contract$rows, length(node_metaclusters) == 100L,
        min(event_metaclusters) >= 1L && max(event_metaclusters) <= 40L,
        all(is.finite(codes)), all(is.finite(bmu_distance)), TRUE
      ),
      detail = c(
        as.character(packageVersion("FlowSOM")), length(node_labels), paste(dim(codes), collapse = "x"),
        paste(range(node_labels), collapse = "-"), length(event_metaclusters), length(node_metaclusters),
        paste(range(event_metaclusters), collapse = "-"), "codes finite", "BMU distances finite", "label column not loaded"
      ), stringsAsFactors = FALSE
    )
    write.csv(checks, file.path(run, "r_checks.csv"), row.names = FALSE)
    manifest <- list(
      experiment_component = "pure_R_FlowSOM_2.18.0_full_event", dataset = dataset, seed = seed,
      completed_utc = format(Sys.time(), tz = "UTC", usetz = TRUE), source = source_path,
      protocol_sha256 = protocol_sha256, package_version = as.character(packageVersion("FlowSOM")),
      r_version = R.version.string, fit_policy = "all_events_direct_training",
      transform = paste0("asinh(x/", cofactor, ")"), scale = FALSE,
      xdim = 10L, ydim = 10L, rlen = 30L, requested_metaclusters = 40L,
      labels_used_for_fit = FALSE, runtime_seconds = runtime,
      checks_passed = sum(checks$passed), checks_total = nrow(checks), all_checks_passed = all(checks$passed)
    )
    jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE, pretty = TRUE)
    if (!all(checks$passed)) stop("R run checks failed")
    new_runs <- new_runs + 1L
    rm(fsom, node_labels, bmu_distance, node_metaclusters, event_metaclusters, codes)
    invisible(gc())
    write_progress()
    cat(dataset, " seed=", seed, " runtime_seconds=", runtime, " checks=10/10\n", sep = "")
    flush.console()
  }, error = function(error) {
    writeLines(c(conditionMessage(error), capture.output(traceback())), file.path(run, "r_failure_traceback.txt"))
    stop(error)
  })
}
write_progress()
cat("R batch complete: ", length(completed_seeds()), "/30\n", sep = "")
