suppressPackageStartupMessages(library(IOBR))
suppressPackageStartupMessages(library(dplyr))

METHOD_ORDER <- c("CIBERSORT", "TIMER", "xCell", "MCPcounter", "ESTIMATE", "EPIC", "IPS", "quanTIseq")

bh_fdr <- function(pvalues) {
  pvalues <- as.numeric(pvalues); out <- rep(NA_real_, length(pvalues)); keep <- is.finite(pvalues)
  if (!any(keep)) return(out)
  order_idx <- order(pvalues[keep]); ranked <- pvalues[keep][order_idx]; adj <- ranked * length(ranked) / seq_along(ranked); adj <- rev(cummin(rev(adj)))
  restored <- numeric(length(adj)); restored[order_idx] <- pmin(pmax(adj, 0), 1); out[which(keep)] <- restored; out
}

read_patient_expression <- function(patient_expr_file) {
  expr <- read.csv(patient_expr_file, row.names = 1, check.names = FALSE); expr <- as.matrix(expr); storage.mode(expr) <- "double"; expr
}

fix_estimate_sample_names <- function(x) gsub("\\.", "-", x)

run_estimate_manual <- function(expr_matrix, output_dir, project = "PUBLIC") {
  expr_df <- as.data.frame(expr_matrix, check.names = FALSE); expr_df$symbol <- rownames(expr_df); expr_df <- expr_df[, c("symbol", setdiff(colnames(expr_df), "symbol"))]
  input_file <- file.path(output_dir, paste0(project, "_estimate_input.txt")); gct_file <- file.path(output_dir, paste0(project, "_estimate_common.gct")); out_file <- file.path(output_dir, paste0(project, "_estimate_score.gct"))
  write.table(expr_df, input_file, sep = "\t", row.names = FALSE, quote = FALSE); filterCommonGenes(input.f = input_file, output.f = gct_file, id = "GeneSymbol"); estimateScore(input.ds = gct_file, output.ds = out_file, platform = "affymetrix")
  scores <- read.table(out_file, skip = 2, header = TRUE, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE); feature_names <- as.character(scores[, 1]); sample_cols <- colnames(scores)[-(1:2)]
  value_mat <- t(as.matrix(scores[, -(1:2), drop = FALSE])); colnames(value_mat) <- paste0(feature_names, "_ESTIMATE"); rownames(value_mat) <- fix_estimate_sample_names(sample_cols)
  invisible(file.remove(input_file)); invisible(file.remove(gct_file)); invisible(file.remove(out_file)); data.frame(ID = rownames(value_mat), value_mat, check.names = FALSE, stringsAsFactors = FALSE)
}

get_method_runner <- function(expr_matrix, output_dir, timer_indications = NULL) {
  if (is.null(timer_indications)) timer_indications <- rep("coad", ncol(expr_matrix))
  list(
    CIBERSORT = function() deconvo_cibersort(expr_matrix, project = "TCGA", arrays = FALSE, perm = 1000, absolute = FALSE),
    TIMER = function() deconvo_timer(expr_matrix, project = "TCGA", indications = timer_indications),
    xCell = function() deconvo_xcell(expr_matrix, project = "TCGA", arrays = FALSE),
    MCPcounter = function() deconvo_mcpcounter(expr_matrix, project = "TCGA"),
    ESTIMATE = function() run_estimate_manual(expr_matrix, output_dir = output_dir, project = "PUBLIC"),
    EPIC = function() deconvo_epic(expr_matrix, project = "TCGA", tumor = TRUE),
    IPS = function() deconvo_ips(expr_matrix, project = "TCGA", plot = FALSE),
    quanTIseq = function() deconvo_quantiseq(expr_matrix, project = "TCGA", tumor = TRUE, arrays = FALSE, scale_mrna = TRUE)
  )
}

prepare_abundance_table <- function(out_df, patient_col = "patient_barcode", id_col = "ID", method_name = NULL) {
  work <- as.data.frame(out_df, stringsAsFactors = FALSE, check.names = FALSE); if (!(id_col %in% colnames(work))) stop("abundance table is missing the sample identifier column")
  work[[patient_col]] <- as.character(work[[id_col]]); feature_cols <- setdiff(colnames(work), c(id_col, patient_col, "ProjectID", "project", "ProjectID_IPS"))
  out <- work[, c(patient_col, feature_cols), drop = FALSE] %>% group_by(.data[[patient_col]]) %>% summarise(across(all_of(feature_cols), ~ mean(as.numeric(.x), na.rm = TRUE)), .groups = "drop")
  if (!is.null(method_name)) { out$Method <- method_name; out <- out[, c("Method", patient_col, feature_cols), drop = FALSE] }
  out
}

correlate_method_with_indices <- function(method_name, abundance_df, patient_indices, patient_col = "patient_barcode", min_pair_n = 8) {
  merged <- inner_join(patient_indices, abundance_df, by = patient_col); index_cols <- setdiff(colnames(patient_indices), patient_col); feature_cols <- setdiff(colnames(abundance_df), c(patient_col, "Method")); rows <- vector("list", length(index_cols) * length(feature_cols)); idx <- 1L
  for (index_name in index_cols) for (feature_name in feature_cols) {
    sub <- merged[, c(index_name, feature_name), drop = FALSE]; sub <- sub[complete.cases(sub), , drop = FALSE]
    if (nrow(sub) < min_pair_n || stats::sd(sub[[index_name]]) == 0 || stats::sd(sub[[feature_name]]) == 0) { rho <- NA_real_; pvalue <- NA_real_ } else { test <- suppressWarnings(stats::cor.test(sub[[index_name]], sub[[feature_name]], method = "spearman", exact = FALSE)); rho <- unname(test$estimate); pvalue <- test$p.value }
    rows[[idx]] <- data.frame(Method = method_name, Index = index_name, Feature = feature_name, SpearmanR = rho, PValue = pvalue, N = nrow(sub), stringsAsFactors = FALSE); idx <- idx + 1L
  }
  out <- bind_rows(rows); out$MethodFDR <- bh_fdr(out$PValue); out %>% arrange(Index, PValue, Feature)
}

build_method_summary <- function(all_corr) all_corr %>% group_by(Method) %>% summarise(Features = n_distinct(Feature), Correlations = n(), SignificantMethodFDR = sum(MethodFDR < 0.05, na.rm = TRUE), TopAbsR = max(abs(SpearmanR), na.rm = TRUE), .groups = "drop")

run_public_iobr_multimethod_workflow <- function(patient_expr_file, patient_indices_file, output_dir, patient_col = "patient_barcode", timer_indications = NULL, methods = METHOD_ORDER) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE); expr_matrix <- read_patient_expression(patient_expr_file); patient_indices <- read.csv(patient_indices_file, stringsAsFactors = FALSE, check.names = FALSE)
  runners <- get_method_runner(expr_matrix, output_dir = output_dir, timer_indications = timer_indications); corr_list <- list()
  for (method_name in methods) {
    out_df <- runners[[method_name]](); write.csv(out_df, file.path(output_dir, paste0("iobr_", method_name, "_abundance.csv")), row.names = FALSE)
    abundance_df <- prepare_abundance_table(out_df, patient_col = patient_col, method_name = method_name); corr_df <- correlate_method_with_indices(method_name, abundance_df, patient_indices, patient_col = patient_col)
    write.csv(corr_df, file.path(output_dir, paste0("iobr_", method_name, "_correlations.csv")), row.names = FALSE); corr_list[[method_name]] <- corr_df
  }
  all_corr <- bind_rows(corr_list); all_corr$GlobalFDR <- bh_fdr(all_corr$PValue); summary_df <- build_method_summary(all_corr)
  write.csv(all_corr, file.path(output_dir, "iobr_multimethod_correlations_all.csv"), row.names = FALSE); write.csv(summary_df, file.path(output_dir, "iobr_multimethod_summary.csv"), row.names = FALSE)
  invisible(list(correlations = all_corr, summary = summary_df))
}
