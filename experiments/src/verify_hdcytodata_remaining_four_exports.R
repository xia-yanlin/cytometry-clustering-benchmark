args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: verify_hdcytodata_remaining_four_exports.R <source_root> <output_dir>")
root <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
output <- normalizePath(args[[2]], winslash = "/", mustWork = FALSE)
dir.create(output, recursive = TRUE, showWarnings = FALSE)
datasets <- c("Levine_32dim", "Samusik_01", "Nilsson_rare", "Mosmann_rare")
rows <- list()
for (dataset in datasets) {
  directory <- file.path(root, dataset)
  object <- readRDS(file.path(directory, paste0(dataset, "_SE_authoritative_extract.rds")))
  expression <- object$expression_untransformed
  connection <- file(file.path(directory, "expression_column_major_f64.bin"), "rb")
  exported <- readBin(connection, what = "double", n = length(expression), size = 8L, endian = "little")
  close(connection)
  exported <- matrix(exported, nrow = nrow(expression), ncol = ncol(expression))
  column_data <- read.csv(file.path(directory, "column_data.csv"), check.names = FALSE, stringsAsFactors = FALSE)
  row_data <- read.csv(file.path(directory, "row_data.csv"), check.names = FALSE, stringsAsFactors = FALSE)
  checks <- c(
    identical(dim(exported), dim(expression)),
    identical(as.double(exported), as.double(expression)),
    identical(as.character(column_data$marker_name), colnames(expression)),
    nrow(row_data) == nrow(expression) && identical(as.character(row_data$population_id), as.character(object$row_data$population_id))
  )
  details <- c(paste(dim(expression), collapse = "x"), paste("exact values", length(expression), "max_abs_delta", max(abs(exported - expression))), paste("columns", ncol(expression)), paste("rows", nrow(expression)))
  names <- c("binary_dimensions", "binary_values_exact", "column_metadata", "row_metadata")
  for (i in seq_along(checks)) rows[[length(rows) + 1L]] <- data.frame(dataset = dataset, check = names[[i]], passed = checks[[i]], detail = details[[i]], stringsAsFactors = FALSE)
}
checks <- do.call(rbind, rows)
write.csv(checks, file.path(output, "r_export_verification_checks.csv"), row.names = FALSE)
capture.output(sessionInfo(), file = file.path(output, "r_session_info.txt"))
cat("checks=", sum(checks$passed), "/", nrow(checks), "\n", sep = "")
if (!all(checks$passed)) quit(status = 1L)
