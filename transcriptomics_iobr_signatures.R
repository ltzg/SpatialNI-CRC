suppressPackageStartupMessages(library(IOBR))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(readr))

bh_fdr <- function(pvalues) {
  pvalues <- as.numeric(pvalues); out <- rep(NA_real_, length(pvalues)); keep <- is.finite(pvalues)
  if (!any(keep)) return(out)
  order_idx <- order(pvalues[keep]); ranked <- pvalues[keep][order_idx]; adj <- ranked * length(ranked) / seq_along(ranked); adj <- rev(cummin(rev(adj)))
  restored <- numeric(length(adj)); restored[order_idx] <- pmin(pmax(adj, 0), 1); out[which(keep)] <- restored; out
}

parse_formula_column <- function(formula_text, frame) eval(parse(text = formula_text), envir = frame)
sanitize_ensembl_id <- function(x) sub("\\..*$", "", as.character(x))

normalize_spatial_index_names <- function(spatial) {
  for (col_name in colnames(spatial)) if (grepl("_has_", col_name) && grepl("_ratio$", col_name)) {
    ni_name <- gsub("_ratio$", "_NI", gsub("_has_", "_to_", col_name)); if (!(ni_name %in% colnames(spatial))) spatial[[ni_name]] <- spatial[[col_name]]
  }
  spatial
}

build_patient_indices <- function(spatial, patient_col, index_def = NULL) {
  spatial <- normalize_spatial_index_names(spatial)
  if (!is.null(index_def)) for (i in seq_len(nrow(index_def))) spatial[[index_def$Index[i]]] <- parse_formula_column(index_def$Formula[i], spatial)
  index_cols <- grep("_NI$", colnames(spatial), value = TRUE)
  if (!is.null(index_def)) index_cols <- unique(c(index_cols, as.character(index_def$Index)))
  spatial %>% select(all_of(c(patient_col, index_cols))) %>% filter(!is.na(.data[[patient_col]]), .data[[patient_col]] != "") %>% group_by(.data[[patient_col]]) %>% summarise(across(all_of(index_cols), ~ mean(as.numeric(.x), na.rm = TRUE)), .groups = "drop") %>% mutate(across(all_of(index_cols), ~ ifelse(is.nan(.x), NA_real_, .x)))
}

load_expression_matrix <- function(expr_file) {
  expr <- fread(expr_file, sep = "\t", data.table = FALSE, showProgress = FALSE, check.names = FALSE)
  colnames(expr)[1] <- "gene_id"; expr$gene_id <- sanitize_ensembl_id(expr$gene_id)
  expr_dt <- as.data.table(expr); numeric_cols <- setdiff(colnames(expr_dt), "gene_id"); expr_dt[, (numeric_cols) := lapply(.SD, as.numeric), .SDcols = numeric_cols]
  as.data.frame(expr_dt[, lapply(.SD, mean, na.rm = TRUE), by = gene_id, .SDcols = numeric_cols], check.names = FALSE)
}

map_gene_symbols <- function(expr, probemap_file = NULL) {
  if (is.null(probemap_file)) {
    out <- expr; rownames(out) <- out$gene_id; out$gene_id <- NULL; return(as.matrix(out))
  }
  probemap <- read_tsv(probemap_file, show_col_types = FALSE) %>% transmute(gene_id = sanitize_ensembl_id(.data[[1]]), gene = as.character(.data[[2]])) %>% distinct()
  out <- expr %>% left_join(probemap, by = "gene_id") %>% mutate(gene = ifelse(is.na(gene) | gene == "", gene_id, gene)) %>% select(-gene_id) %>% group_by(gene) %>% summarise(across(everything(), ~ mean(as.numeric(.x), na.rm = TRUE)), .groups = "drop")
  out <- as.data.frame(out, check.names = FALSE); rownames(out) <- out$gene; out$gene <- NULL; as.matrix(out)
}

ensure_log_scale <- function(expr_matrix) {
  storage.mode(expr_matrix) <- "double"; q99 <- suppressWarnings(as.numeric(quantile(expr_matrix, probs = 0.99, na.rm = TRUE)))
  if (is.na(q99) || q99 < 30) expr_matrix else log2(expr_matrix + 1)
}

build_patient_expression <- function(expr_matrix, sample_meta, sample_col = "sample_id", patient_col = "patient_barcode", tumor_col = "is_tumor") {
  if (tumor_col %in% colnames(sample_meta)) sample_meta <- sample_meta %>% filter(.data[[tumor_col]])
  sample_meta <- sample_meta %>% filter(.data[[sample_col]] %in% colnames(expr_matrix))
  sample_groups <- split(sample_meta[[sample_col]], sample_meta[[patient_col]])
  out <- lapply(sample_groups, function(samples) if (length(samples) == 1) expr_matrix[, samples, drop = TRUE] else rowMeans(expr_matrix[, samples, drop = FALSE], na.rm = TRUE))
  out <- do.call(cbind, out); colnames(out) <- names(sample_groups); out
}

run_iobr_signature_scores <- function(expr_matrix, score_method = c("zscore", "ssgsea"), signature_object = signature_tme) {
  score_method <- match.arg(score_method)
  score_tbl <- if (score_method == "ssgsea") calculate_sig_score(eset = expr_matrix, signature = signature_object, method = score_method, mini_gene_count = 3, parallel.size = 1) else calculate_sig_score(eset = expr_matrix, signature = signature_object, method = score_method, mini_gene_count = 3)
  as.data.frame(score_tbl, check.names = FALSE)
}

correlate_scores_with_indices <- function(score_df, analysis_df, patient_col = "patient_barcode", min_pair_n = 8) {
  merged <- inner_join(analysis_df, score_df, by = patient_col); index_cols <- setdiff(colnames(analysis_df), patient_col); feature_cols <- setdiff(colnames(score_df), patient_col); rows <- vector("list", length(index_cols) * length(feature_cols)); idx <- 1L
  for (index_name in index_cols) for (feature_name in feature_cols) {
    sub <- merged[, c(index_name, feature_name), drop = FALSE]; sub <- sub[complete.cases(sub), , drop = FALSE]
    if (nrow(sub) < min_pair_n || stats::sd(sub[[index_name]]) == 0 || stats::sd(sub[[feature_name]]) == 0) { rho <- NA_real_; pvalue <- NA_real_ } else { test <- suppressWarnings(cor.test(sub[[index_name]], sub[[feature_name]], method = "spearman", exact = FALSE)); rho <- unname(test$estimate); pvalue <- test$p.value }
    rows[[idx]] <- data.frame(Index = index_name, Feature = feature_name, SpearmanR = rho, PValue = pvalue, N = nrow(sub), stringsAsFactors = FALSE); idx <- idx + 1L
  }
  out <- bind_rows(rows); out$FDR <- bh_fdr(out$PValue); out %>% arrange(Index, PValue, Feature)
}

run_public_iobr_signature_workflow <- function(spatial_file, expression_file, sample_meta_file, output_dir, patient_col = "patient_barcode", index_def_file = NULL, probemap_file = NULL, score_method = c("zscore", "ssgsea")) {
  score_method <- match.arg(score_method); dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  spatial <- read.csv(spatial_file, stringsAsFactors = FALSE, check.names = FALSE); index_def <- if (is.null(index_def_file)) NULL else read.csv(index_def_file, stringsAsFactors = FALSE, check.names = FALSE)
  patient_indices <- build_patient_indices(spatial, patient_col = patient_col, index_def = index_def)
  expr_log <- ensure_log_scale(map_gene_symbols(load_expression_matrix(expression_file), probemap_file = probemap_file)); sample_meta <- read.csv(sample_meta_file, stringsAsFactors = FALSE, check.names = FALSE)
  patient_expr <- build_patient_expression(expr_log, sample_meta); score_df <- run_iobr_signature_scores(patient_expr, score_method = score_method); colnames(score_df)[1] <- patient_col
  matched <- intersect(colnames(patient_expr), patient_indices[[patient_col]]); matched <- matched[order(matched)]
  patient_indices <- patient_indices %>% filter(.data[[patient_col]] %in% matched) %>% arrange(match(.data[[patient_col]], matched)); score_df <- score_df %>% filter(.data[[patient_col]] %in% matched) %>% arrange(match(.data[[patient_col]], matched))
  corr_df <- correlate_scores_with_indices(score_df, patient_indices, patient_col = patient_col)
  write.csv(score_df, file.path(output_dir, paste0("iobr_signature_scores_", score_method, ".csv")), row.names = FALSE); write.csv(corr_df, file.path(output_dir, paste0("iobr_signature_correlations_", score_method, ".csv")), row.names = FALSE)
  invisible(list(indices = patient_indices, scores = score_df, correlations = corr_df))
}
