args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop("usage: retrieve_hdcytodata_levine13_authoritative.R <target_library> <shared_library> <cache_dir> <output_dir>")
}

target_library <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
shared_library <- normalizePath(args[[2]], winslash = "/", mustWork = TRUE)
cache_dir <- normalizePath(args[[3]], winslash = "/", mustWork = FALSE)
output_dir <- normalizePath(args[[4]], winslash = "/", mustWork = FALSE)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(target_library, shared_library, .Library))

suppressPackageStartupMessages({
  library(ExperimentHub)
  library(HDCytoData)
  library(SummarizedExperiment)
})

setExperimentHubOption("CACHE", cache_dir)
setExperimentHubOption("ASK", FALSE)

metadata_record <- HDCytoData::Levine_13dim_SE(metadata = TRUE)
if (length(metadata_record) != 1L) {
  stop("expected exactly one Levine_13dim_SE ExperimentHub record")
}
resource_id <- names(metadata_record)[[1L]]
metadata_table <- as.data.frame(S4Vectors::mcols(metadata_record), stringsAsFactors = FALSE)
metadata_table <- cbind(resource_id = resource_id, metadata_table, stringsAsFactors = FALSE)
write.csv(
  metadata_table,
  file.path(output_dir, "experimenthub_metadata.csv"),
  row.names = FALSE,
  na = ""
)

object <- HDCytoData::Levine_13dim_SE(metadata = FALSE)
exprs_matrix <- as.matrix(SummarizedExperiment::assay(object, "exprs"))
population_id <- as.character(SummarizedExperiment::rowData(object)$population_id)
marker_info <- as.data.frame(SummarizedExperiment::colData(object), stringsAsFactors = FALSE)

if (!identical(dim(exprs_matrix), c(167044L, 13L))) {
  stop("unexpected authoritative expression dimensions: ", paste(dim(exprs_matrix), collapse = "x"))
}
if (length(population_id) != nrow(exprs_matrix)) {
  stop("population vector length does not match expression rows")
}

saveRDS(
  list(
    resource_id = resource_id,
    expression_untransformed = exprs_matrix,
    population_id = population_id,
    marker_info = marker_info
  ),
  file.path(output_dir, "Levine_13dim_SE_authoritative_extract.rds"),
  version = 3,
  compress = "xz"
)

write.csv(marker_info, file.path(output_dir, "authoritative_marker_info.csv"), row.names = FALSE, na = "")
write.csv(
  as.data.frame(sort(table(population_id), decreasing = TRUE), stringsAsFactors = FALSE),
  file.path(output_dir, "authoritative_population_counts.csv"),
  row.names = FALSE
)

summary_table <- data.frame(
  field = c(
    "resource_id", "rows", "expression_columns", "unique_population_values",
    "unassigned_events", "assigned_events", "package_HDCytoData",
    "package_ExperimentHub", "package_SummarizedExperiment", "cache_dir"
  ),
  value = c(
    resource_id,
    nrow(exprs_matrix),
    ncol(exprs_matrix),
    length(unique(population_id)),
    sum(population_id == "unassigned"),
    sum(population_id != "unassigned"),
    as.character(packageVersion("HDCytoData")),
    as.character(packageVersion("ExperimentHub")),
    as.character(packageVersion("SummarizedExperiment")),
    cache_dir
  ),
  stringsAsFactors = FALSE
)
write.csv(summary_table, file.path(output_dir, "authoritative_object_summary.csv"), row.names = FALSE)

cat("resource_id=", resource_id, "\n", sep = "")
cat("dimensions=", nrow(exprs_matrix), "x", ncol(exprs_matrix), "\n", sep = "")
cat("assigned=", sum(population_id != "unassigned"), "\n", sep = "")
cat("unassigned=", sum(population_id == "unassigned"), "\n", sep = "")
