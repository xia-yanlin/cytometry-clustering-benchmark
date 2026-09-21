args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("usage: compare_hdcytodata_levine13_numeric_identity.R <authoritative_extract.rds> <local_txt> <output_dir>")
}

authoritative_path <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
local_path <- normalizePath(args[[2]], winslash = "/", mustWork = TRUE)
output_dir <- normalizePath(args[[3]], winslash = "/", mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

authoritative <- readRDS(authoritative_path)
auth_expr <- authoritative$expression_untransformed
auth_population <- authoritative$population_id
auth_markers <- colnames(auth_expr)

local <- read.delim(local_path, header = TRUE, check.names = FALSE, stringsAsFactors = FALSE)
local_markers <- colnames(local)[seq_len(13L)]
local_expr <- as.matrix(local[, seq_len(13L), drop = FALSE])
storage.mode(local_expr) <- "double"
local_label <- suppressWarnings(as.integer(local[[14L]]))

transformed <- asinh(auth_expr / 5)
delta <- transformed - local_expr
abs_delta <- abs(delta)
raw_abs_delta <- abs(auth_expr - local_expr)
inverse_abs_delta <- abs(5 * sinh(local_expr) - auth_expr)
tolerance <- 5e-5

quantile_names <- c("q0", "q50", "q90", "q99", "q999", "q100")
quantile_values <- as.numeric(quantile(abs_delta, probs = c(0, .5, .9, .99, .999, 1), names = FALSE, type = 8))
summary_rows <- data.frame(
  metric = c(
    "elements", "max_abs_delta_transformed", "mean_abs_delta_transformed",
    quantile_names, "fraction_exact_double", "fraction_within_1e-12",
    "fraction_within_1e-8", "fraction_within_5e-5", "max_abs_delta_raw_vs_local",
    "mean_abs_delta_raw_vs_local", "max_abs_delta_inverse_to_raw",
    "mean_abs_delta_inverse_to_raw"
  ),
  value = c(
    length(abs_delta), max(abs_delta), mean(abs_delta), quantile_values,
    mean(delta == 0), mean(abs_delta <= 1e-12), mean(abs_delta <= 1e-8),
    mean(abs_delta <= tolerance), max(raw_abs_delta), mean(raw_abs_delta),
    max(inverse_abs_delta), mean(inverse_abs_delta)
  ),
  stringsAsFactors = FALSE
)
write.csv(summary_rows, file.path(output_dir, "numeric_identity_summary.csv"), row.names = FALSE)

per_marker <- do.call(rbind, lapply(seq_len(ncol(local_expr)), function(j) {
  d <- abs_delta[, j]
  inv <- inverse_abs_delta[, j]
  data.frame(
    marker_index = j,
    authoritative_marker = auth_markers[[j]],
    local_marker = local_markers[[j]],
    max_abs_delta = max(d),
    mean_abs_delta = mean(d),
    q99_abs_delta = as.numeric(quantile(d, .99, names = FALSE, type = 8)),
    exact_fraction = mean(d == 0),
    within_1e_12_fraction = mean(d <= 1e-12),
    within_5e_5_fraction = mean(d <= tolerance),
    max_inverse_abs_delta = max(inv),
    stringsAsFactors = FALSE
  )
}))
write.csv(per_marker, file.path(output_dir, "per_marker_numeric_identity.csv"), row.names = FALSE)

assigned <- !is.na(local_label)
mapping <- unique(data.frame(
  local_label = local_label[assigned],
  authoritative_population = auth_population[assigned],
  stringsAsFactors = FALSE
))
mapping <- mapping[order(mapping$local_label, mapping$authoritative_population), , drop = FALSE]
mapping$count <- vapply(seq_len(nrow(mapping)), function(i) {
  sum(local_label == mapping$local_label[[i]] & auth_population == mapping$authoritative_population[[i]], na.rm = TRUE)
}, integer(1))
mapping$first_row_1based <- vapply(seq_len(nrow(mapping)), function(i) {
  which(local_label == mapping$local_label[[i]] & auth_population == mapping$authoritative_population[[i]])[[1L]]
}, integer(1))
mapping$last_row_1based <- vapply(seq_len(nrow(mapping)), function(i) {
  tail(which(local_label == mapping$local_label[[i]] & auth_population == mapping$authoritative_population[[i]]), 1L)
}, integer(1))
write.csv(mapping, file.path(output_dir, "label_mapping.csv"), row.names = FALSE)

check_rows <- list()
add_check <- function(name, passed, detail) {
  check_rows[[length(check_rows) + 1L]] <<- data.frame(
    check = name, passed = isTRUE(passed), detail = as.character(detail), stringsAsFactors = FALSE
  )
}
add_check("authoritative_dimensions", identical(dim(auth_expr), c(167044L, 13L)), paste(dim(auth_expr), collapse = "x"))
add_check("local_dimensions", identical(dim(local), c(167044L, 14L)), paste(dim(local), collapse = "x"))
add_check("marker_names_and_order_exact", identical(auth_markers, local_markers), paste(local_markers, collapse = "|"))
add_check("all_expression_values_finite", all(is.finite(auth_expr)) && all(is.finite(local_expr)), "authoritative and local matrices finite")
add_check("local_label_contract", sum(assigned) == 81747L && sum(!assigned) == 85297L && identical(sort(unique(local_label[assigned])), 1:24), paste(sum(assigned), sum(!assigned), sep = "/"))
add_check("authoritative_label_contract", sum(auth_population != "unassigned") == 81747L && sum(auth_population == "unassigned") == 85297L && length(setdiff(unique(auth_population), "unassigned")) == 24L, paste(sum(auth_population != "unassigned"), sum(auth_population == "unassigned"), sep = "/"))
add_check("unassigned_rows_exact", identical(is.na(local_label), auth_population == "unassigned"), paste(sum(is.na(local_label) == (auth_population == "unassigned")), "rows"))
add_check("one_to_one_label_mapping", nrow(mapping) == 24L && length(unique(mapping$local_label)) == 24L && length(unique(mapping$authoritative_population)) == 24L, paste(nrow(mapping), "mapping rows"))
add_check("transformed_numeric_identity_tolerance", all(abs_delta <= tolerance), paste("max", format(max(abs_delta), digits = 17), "tolerance", tolerance))
add_check("transformed_identity_strict_1e12", all(abs_delta <= 1e-12), paste("max", format(max(abs_delta), digits = 17)))
add_check("raw_matrix_is_not_local_matrix", max(raw_abs_delta) > 1 && mean(raw_abs_delta) > 1, paste("max", format(max(raw_abs_delta), digits = 12), "mean", format(mean(raw_abs_delta), digits = 12)))
add_check("inverse_transform_recovers_authoritative", all(inverse_abs_delta <= 1e-8), paste("max", format(max(inverse_abs_delta), digits = 17)))
checks <- do.call(rbind, check_rows)
write.csv(checks, file.path(output_dir, "numeric_identity_checks.csv"), row.names = FALSE)

capture.output(sessionInfo(), file = file.path(output_dir, "r_session_info.txt"))
cat("checks=", sum(checks$passed), "/", nrow(checks), "\n", sep = "")
cat("max_abs_delta=", format(max(abs_delta), digits = 17), "\n", sep = "")
cat("within_5e-5_fraction=", format(mean(abs_delta <= tolerance), digits = 17), "\n", sep = "")
cat("mapping_rows=", nrow(mapping), "\n", sep = "")

if (!all(checks$passed)) quit(status = 1L)
