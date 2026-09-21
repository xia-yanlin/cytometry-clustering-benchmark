args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop("usage: verify_hdcytodata_levine13_numeric_identity.R <authoritative_extract.rds> <local_txt> <parent_run> <output_dir>")
}

authoritative <- readRDS(normalizePath(args[[1]], winslash = "/", mustWork = TRUE))
local <- read.delim(normalizePath(args[[2]], winslash = "/", mustWork = TRUE), check.names = FALSE, stringsAsFactors = FALSE)
parent <- normalizePath(args[[3]], winslash = "/", mustWork = TRUE)
output <- normalizePath(args[[4]], winslash = "/", mustWork = FALSE)
dir.create(output, recursive = TRUE, showWarnings = FALSE)

auth_expr <- authoritative$expression_untransformed
local_expr <- as.matrix(local[, 1:13, drop = FALSE])
storage.mode(local_expr) <- "double"
auth_population <- authoritative$population_id
local_label <- suppressWarnings(as.integer(local[[14L]]))
delta <- abs(asinh(auth_expr / 5) - local_expr)

reported_summary <- read.csv(file.path(parent, "numeric_identity_summary.csv"), stringsAsFactors = FALSE)
reported <- setNames(reported_summary$value, reported_summary$metric)
reported_marker <- read.csv(file.path(parent, "per_marker_numeric_identity.csv"), stringsAsFactors = FALSE, check.names = FALSE)
reported_mapping <- read.csv(file.path(parent, "label_mapping.csv"), stringsAsFactors = FALSE, check.names = FALSE)

independent_summary <- data.frame(
  metric = c("elements", "max_abs_delta_transformed", "mean_abs_delta_transformed", "fraction_exact_double", "fraction_within_1e-12", "fraction_within_5e-5"),
  value = c(length(delta), max(delta), mean(delta), mean(delta == 0), mean(delta <= 1e-12), mean(delta <= 5e-5)),
  stringsAsFactors = FALSE
)
write.csv(independent_summary, file.path(output, "independent_numeric_summary.csv"), row.names = FALSE)

independent_marker <- do.call(rbind, lapply(1:13, function(j) data.frame(
  marker_index = j,
  authoritative_marker = colnames(auth_expr)[[j]],
  local_marker = colnames(local)[[j]],
  max_abs_delta = max(delta[, j]),
  mean_abs_delta = mean(delta[, j]),
  stringsAsFactors = FALSE
)))
write.csv(independent_marker, file.path(output, "independent_per_marker_summary.csv"), row.names = FALSE)

assigned <- !is.na(local_label)
independent_mapping <- unique(data.frame(
  local_label = local_label[assigned],
  authoritative_population = auth_population[assigned],
  stringsAsFactors = FALSE
))
independent_mapping <- independent_mapping[order(independent_mapping$local_label, independent_mapping$authoritative_population), , drop = FALSE]
independent_mapping$count <- vapply(seq_len(nrow(independent_mapping)), function(i) sum(local_label == independent_mapping$local_label[[i]] & auth_population == independent_mapping$authoritative_population[[i]], na.rm = TRUE), integer(1))
independent_mapping$first_row_1based <- vapply(seq_len(nrow(independent_mapping)), function(i) which(local_label == independent_mapping$local_label[[i]] & auth_population == independent_mapping$authoritative_population[[i]])[[1L]], integer(1))
independent_mapping$last_row_1based <- vapply(seq_len(nrow(independent_mapping)), function(i) tail(which(local_label == independent_mapping$local_label[[i]] & auth_population == independent_mapping$authoritative_population[[i]]), 1L), integer(1))
write.csv(independent_mapping, file.path(output, "independent_label_mapping.csv"), row.names = FALSE)

rows <- list()
add <- function(name, passed, detail) rows[[length(rows) + 1L]] <<- data.frame(check = name, passed = isTRUE(passed), detail = as.character(detail), stringsAsFactors = FALSE)
close_num <- function(a, b, tol = 1e-15) isTRUE(all(abs(as.numeric(a) - as.numeric(b)) <= tol))
add("dimensions_independent", identical(dim(auth_expr), c(167044L, 13L)) && identical(dim(local), c(167044L, 14L)), paste(dim(auth_expr), dim(local), collapse = "/"))
add("markers_independent", identical(colnames(auth_expr), colnames(local)[1:13]), paste(colnames(auth_expr), collapse = "|"))
add("strict_numeric_identity", max(delta) <= 1e-12 && length(delta) == 2171572L, format(max(delta), digits = 17))
add("reported_global_values", close_num(reported[["max_abs_delta_transformed"]], max(delta)) && close_num(reported[["mean_abs_delta_transformed"]], mean(delta)) && close_num(reported[["fraction_exact_double"]], mean(delta == 0)) && close_num(reported[["fraction_within_1e-12"]], mean(delta <= 1e-12)), "parent global values reproduced")
add("reported_tolerance_fraction", close_num(reported[["fraction_within_5e-5"]], 1) && all(delta <= 5e-5), "all elements within preregistered tolerance")
add("per_marker_names", identical(as.character(reported_marker$authoritative_marker), colnames(auth_expr)) && identical(as.character(reported_marker$local_marker), colnames(local)[1:13]), "13 marker names and order")
add("per_marker_values", close_num(reported_marker$max_abs_delta, independent_marker$max_abs_delta) && close_num(reported_marker$mean_abs_delta, independent_marker$mean_abs_delta), "all 26 numeric summaries reproduced")
add("unassigned_rows_independent", identical(is.na(local_label), auth_population == "unassigned"), "167044 rowwise statuses")
add("label_mapping_independent", isTRUE(all.equal(reported_mapping, independent_mapping, check.attributes = FALSE, tolerance = 0)), paste(nrow(independent_mapping), "rows"))
add("population_counts_independent", sum(!is.na(local_label)) == 81747L && sum(is.na(local_label)) == 85297L && nrow(independent_mapping) == 24L, "81747 assigned, 85297 unassigned, 24 mappings")
checks <- do.call(rbind, rows)
write.csv(checks, file.path(output, "independent_numeric_checks.csv"), row.names = FALSE)
capture.output(sessionInfo(), file = file.path(output, "r_session_info.txt"))
cat("checks=", sum(checks$passed), "/", nrow(checks), "\n", sep = "")
cat("max_abs_delta=", format(max(delta), digits = 17), "\n", sep = "")
if (!all(checks$passed)) quit(status = 1L)
