args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop("usage: retrieve_hdcytodata_remaining_four.R <target_library> <shared_library> <cache_dir> <output_root>")
}

target_library <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
shared_library <- normalizePath(args[[2]], winslash = "/", mustWork = TRUE)
cache_dir <- normalizePath(args[[3]], winslash = "/", mustWork = FALSE)
output_root <- normalizePath(args[[4]], winslash = "/", mustWork = FALSE)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(target_library, shared_library, .Library))

suppressPackageStartupMessages({
  library(ExperimentHub)
  library(HDCytoData)
  library(SummarizedExperiment)
})
setExperimentHubOption("CACHE", cache_dir)
setExperimentHubOption("ASK", FALSE)

datasets <- c("Levine_32dim", "Samusik_01", "Nilsson_rare", "Mosmann_rare")
for (dataset in datasets) {
  accessor <- getExportedValue("HDCytoData", paste0(dataset, "_SE"))
  metadata_record <- accessor(metadata = TRUE)
  if (length(metadata_record) != 1L) stop(dataset, ": expected exactly one ExperimentHub record")
  resource_id <- names(metadata_record)[[1L]]
  object <- accessor(metadata = FALSE)
  expression <- as.matrix(SummarizedExperiment::assay(object, "exprs"))
  row_data <- as.data.frame(SummarizedExperiment::rowData(object), stringsAsFactors = FALSE)
  column_data <- as.data.frame(SummarizedExperiment::colData(object), stringsAsFactors = FALSE)
  dataset_dir <- file.path(output_root, dataset)
  dir.create(dataset_dir, recursive = TRUE, showWarnings = FALSE)
  metadata_table <- cbind(resource_id = resource_id, as.data.frame(S4Vectors::mcols(metadata_record), stringsAsFactors = FALSE), stringsAsFactors = FALSE)
  write.csv(metadata_table, file.path(dataset_dir, "experimenthub_metadata.csv"), row.names = FALSE, na = "")
  write.csv(row_data, file.path(dataset_dir, "row_data.csv"), row.names = FALSE, na = "")
  write.csv(column_data, file.path(dataset_dir, "column_data.csv"), row.names = FALSE, na = "")
  saveRDS(
    list(resource_id = resource_id, expression_untransformed = expression, row_data = row_data, column_data = column_data),
    file.path(dataset_dir, paste0(dataset, "_SE_authoritative_extract.rds")),
    version = 3, compress = "xz"
  )
  # A base-R binary interchange lets Python compare the original Chinese-path
  # project files without duplicating them solely for R path compatibility.
  connection <- file(file.path(dataset_dir, "expression_column_major_f64.bin"), open = "wb")
  writeBin(as.double(expression), connection, size = 8L, endian = "little")
  close(connection)
  write.csv(
    data.frame(field = c("dataset", "resource_id", "rows", "columns", "binary_order", "binary_type"), value = c(dataset, resource_id, nrow(expression), ncol(expression), "column-major", "little-endian float64"), stringsAsFactors = FALSE),
    file.path(dataset_dir, "authoritative_object_summary.csv"), row.names = FALSE
  )
  cat(dataset, " resource_id=", resource_id, " dimensions=", nrow(expression), "x", ncol(expression), "\n", sep = "")
}

capture.output(sessionInfo(), file = file.path(output_root, "retrieval_session_info.txt"))
