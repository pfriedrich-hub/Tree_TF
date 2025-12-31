#!/usr/bin/env Rscript

# =============================================================================
# Stepwise Modeling for Tree Sound Absorption
# =============================================================================

# Performs stepwise forward selection with multiple criteria.
# Analyzes: Broadleaf, Needleleaf, AND Mixed (all trees combined)
#
# Usage:
# Rscript run_stepwise_modeling.R --data output/csv/merged_data.csv --output output/figures/models

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(optparse)
  library(patchwork)
})

USE_REPEL <- requireNamespace("ggrepel", quietly = TRUE)
if (USE_REPEL) library(ggrepel)

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_OUTPUT_DIR <- "output/figures/models"

RESPONSE_VARS <- c(
  "attenuation_low_db",
  "attenuation_mid_db",
  "attenuation_high_db",
  "attenuation_overall_db"
)

# Predictors for analysis (precursors removed: densest_slice_height, distance,
# pts_per_voxel, dry/fresh weight)
PREDICTORS <- c(
  "height_m", "crown_volume_m3", "vol_fill_fraction", "gap_fraction",
  "leaf_voxel_density", "ENL", "FHD", "LAD", "leaf_area_cm2",
  "leaf_thickness_mm", "ldmc", "toughness"
)

CRITERIA_LIST <- list(
  adj_r2 = list(name = "adj_r2", label = "Adjusted R² (≥2%)", threshold = 0.02),
  f_test = list(name = "f_test", label = "F-test (p < 0.15)", threshold = 0.15),
  loocv  = list(name = "loocv",  label = "LOOCV",           threshold = 0),
  bic    = list(name = "bic",    label = "BIC",             threshold = 0)
)

VAR_LABELS <- c(
  "height_m"           = "Height",
  "crown_volume_m3"    = "Crown Vol.",
  "vol_fill_fraction"  = "Vol. Fill",
  "gap_fraction"       = "Gap Frac.",
  "leaf_voxel_density" = "Voxel Dens.",
  "ENL"                = "ENL",
  "FHD"                = "FHD",
  "LAD"                = "LAD",
  "leaf_area_cm2"      = "Leaf Area",
  "leaf_thickness_mm"  = "Thickness",
  "ldmc"               = "LDMC",
  "toughness"          = "Toughness"
)

# Color scheme: red(broad) + blue(needle) = purple(mixed)
CB_ROSE   <- "#CC6677" # Broadleaf (reddish)
CB_INDIGO <- "#332288" # Needleleaf (bluish)
CB_PURPLE <- "#AA4499" # Mixed analysis (purple)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

loocv_rmse <- function(model, data, response) {
  preds <- numeric(nrow(data))
  for (i in 1:nrow(data)) {
    m <- lm(formula(model), data = data[-i, ])
    preds[i] <- predict(m, newdata = data[i, ])
  }
  sqrt(mean((data[[response]] - preds)^2))
}

analyze_collinearity <- function(df, predictors, output_dir, leaf_type) {
  avail <- predictors[predictors %in% names(df)]
  df_preds <- df[, ..avail]
  df_complete <- na.omit(df_preds)
  if (nrow(df_complete) < 5) return(NULL)

  cor_mat <- cor(df_complete, use = "pairwise.complete.obs")
  cor_df <- as.data.frame(as.table(cor_mat))
  names(cor_df) <- c("Var1", "Var2", "r")

  cor_df$Var1_lab <- ifelse(
    as.character(cor_df$Var1) %in% names(VAR_LABELS),
    VAR_LABELS[as.character(cor_df$Var1)],
    as.character(cor_df$Var1)
  )
  cor_df$Var2_lab <- ifelse(
    as.character(cor_df$Var2) %in% names(VAR_LABELS),
    VAR_LABELS[as.character(cor_df$Var2)],
    as.character(cor_df$Var2)
  )

  p <- ggplot(cor_df, aes(x = Var1_lab, y = Var2_lab, fill = r)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", r)), size = 2.5) +
    scale_fill_gradient2(
      low = "#2166AC", mid = "white", high = "#B2182B",
      midpoint = 0, limits = c(-1, 1)
    ) +
    labs(
      title = sprintf("Predictor Correlations: %s", leaf_type),
      x = "", y = ""
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(face = "bold")
    )

  ggsave(
    file.path(
      output_dir,
      sprintf("predictor_correlations_%s.png", tolower(gsub(" ", "_", leaf_type)))
    ),
    p, width = 9, height = 8, dpi = 200
  )

  return(cor_mat)
}

# ============================================================
# FORWARD STEPWISE SELECTION
# ============================================================

forward_stepwise <- function(df, response, predictors,
                             max_preds = 3,
                             criterion = "adj_r2",
                             log_file = NULL) {
  log_msg <- function(msg) {
    message(msg)
    if (!is.null(log_file)) cat(msg, "\n", file = log_file, append = TRUE)
  }

  avail <- predictors[predictors %in% names(df)]
  vars_needed <- c(response, avail)
  df_complete <- na.omit(df[, ..vars_needed])
  n <- nrow(df_complete)

  log_msg(sprintf("\n--- Forward Stepwise for %s (n=%d) ---", response, n))
  if (n < 5) {
    log_msg(" ERROR: Not enough cases")
    return(NULL)
  }

  current_preds <- character(0)
  remaining <- avail
  steps <- list()

  null_model <- lm(as.formula(sprintf("%s ~ 1", response)), data = df_complete)
  current_adj_r2 <- 0
  current_bic <- BIC(null_model)
  current_loocv <- loocv_rmse(null_model, df_complete, response)
  step_num <- 0

  while (length(current_preds) < max_preds && length(remaining) > 0) {
    step_num <- step_num + 1

    if (length(current_preds) == 0) {
      current_model <- null_model
    } else {
      current_model <- lm(
        as.formula(sprintf("%s ~ %s", response, paste(current_preds, collapse = " + "))),
        data = df_complete
      )
    }

    candidates <- data.frame(
      predictor    = character(),
      adj_r2       = numeric(),
      delta_adj_r2 = numeric(),
      bic          = numeric(),
      delta_bic    = numeric(),
      loocv_rmse   = numeric(),
      delta_loocv  = numeric(),
      f_pvalue     = numeric(),
      stringsAsFactors = FALSE
    )

    for (pred in remaining) {
      test_preds <- c(current_preds, pred)
      tryCatch({
        test_model <- lm(
          as.formula(sprintf("%s ~ %s", response, paste(test_preds, collapse = " + "))),
          data = df_complete
        )
        test_adj_r2  <- summary(test_model)$adj.r.squared
        test_bic     <- BIC(test_model)
        test_loocv   <- loocv_rmse(test_model, df_complete, response)
        f_result     <- anova(current_model, test_model)
        f_pvalue     <- f_result$`Pr(>F)`[2]
        if (is.na(f_pvalue)) f_pvalue <- 1

        candidates <- rbind(
          candidates,
          data.frame(
            predictor    = pred,
            adj_r2       = test_adj_r2,
            delta_adj_r2 = test_adj_r2 - current_adj_r2,
            bic          = test_bic,
            delta_bic    = test_bic - current_bic,
            loocv_rmse   = test_loocv,
            delta_loocv  = test_loocv - current_loocv,
            f_pvalue     = f_pvalue,
            stringsAsFactors = FALSE
          )
        )
      }, error = function(e) NULL)
    }

    if (nrow(candidates) == 0) break

    if (criterion == "adj_r2")      candidates <- candidates[order(-candidates$adj_r2), ]
    else if (criterion == "f_test") candidates <- candidates[order(candidates$f_pvalue), ]
    else if (criterion == "loocv")  candidates <- candidates[order(candidates$loocv_rmse), ]
    else if (criterion == "bic")    candidates <- candidates[order(candidates$bic), ]

    best <- candidates[1, ]
    select_pred <- FALSE

    if (criterion == "adj_r2" && best$delta_adj_r2 >= 0.02) select_pred <- TRUE
    if (criterion == "f_test" && best$f_pvalue < 0.15)       select_pred <- TRUE
    if (criterion == "loocv"  && best$delta_loocv < 0)       select_pred <- TRUE
    if (criterion == "bic"    && best$delta_bic < 0)         select_pred <- TRUE

    if (!select_pred) {
      log_msg(sprintf(" Step %d: STOP - no improvement", step_num))
      break
    }

    log_msg(sprintf(" Step %d: +%s (Adj.R²=%.3f)", step_num, best$predictor, best$adj_r2))

    current_preds <- c(current_preds, best$predictor)
    remaining     <- setdiff(remaining, best$predictor)
    current_adj_r2 <- best$adj_r2
    current_bic    <- best$bic
    current_loocv  <- best$loocv_rmse

    # Store full candidate table for plotting
    steps[[step_num]] <- list(
      added          = best$predictor,
      adj_r2         = best$adj_r2,
      all_candidates = candidates
    )
  }

  if (length(current_preds) == 0) {
    log_msg(" FINAL: Null model")
    return(list(
      final_preds = character(0),
      final_model = null_model,
      steps       = steps,
      n           = n
    ))
  }

  final_model <- lm(
    as.formula(sprintf("%s ~ %s", response, paste(current_preds, collapse = " + "))),
    data = df_complete
  )
  log_msg(sprintf(
    " FINAL: %s (Adj.R²=%.3f)",
    paste(current_preds, collapse = " + "),
    summary(final_model)$adj.r.squared
  ))

  return(list(
    final_preds = current_preds,
    final_model = final_model,
    steps       = steps,
    n           = n
  ))
}

# ============================================================
# VISUALIZATION
# ============================================================

create_summary_heatmap <- function(results_broad, results_needle, results_mixed,
                                   output_dir, criteria_name, criteria_label) {
  final_rows <- list()

  add_finals <- function(results, leaf_type_name) {
    for (resp_name in names(results)) {
      res <- results[[resp_name]]
      if (is.null(res)) next
      resp_short <- gsub("attenuation_|_db", "", resp_name)

      if (length(res$final_preds) > 0) {
        adj_r2_val <- summary(res$final_model)$adj.r.squared
        preds_str  <- paste(res$final_preds, collapse = " + ")
      } else {
        adj_r2_val <- 0
        preds_str  <- "(null)"
      }

      final_rows[[length(final_rows) + 1]] <- data.frame(
        leaf_type  = leaf_type_name,
        response   = resp_short,
        predictors = preds_str,
        adj_r2     = adj_r2_val,
        stringsAsFactors = FALSE
      )
    }
  }

  add_finals(results_broad,  "Broadleaf")
  add_finals(results_needle, "Needleleaf")
  add_finals(results_mixed,  "Mixed")

  if (length(final_rows) == 0) return()

  df_final <- do.call(rbind, final_rows)
  df_final$response <- factor(
    df_final$response,
    levels = c("low", "mid", "high", "overall"),
    labels = c("Low", "Mid", "High", "Overall")
  )
  df_final$leaf_type <- factor(
    df_final$leaf_type,
    levels = c("Broadleaf", "Needleleaf", "Mixed")
  )

  df_final$pred_nice <- sapply(df_final$predictors, function(x) {
    if (x == "(null)") return("(null)")
    preds <- strsplit(x, " \\+ ")[[1]]
    paste(
      sapply(preds, function(p) {
        if (p %in% names(VAR_LABELS)) VAR_LABELS[p] else p
      }),
      collapse = "\n+ "
    )
  })

  df_final$label <- sprintf("%s\n(%.2f)", df_final$pred_nice, df_final$adj_r2)

  p <- ggplot(df_final, aes(x = response, y = leaf_type, fill = adj_r2)) +
    geom_tile(color = "white", linewidth = 1.5) +
    geom_text(aes(label = label), size = 2.8, lineheight = 0.85) +
    scale_fill_gradient(
      low = "white", high = "#2166AC",
      limits = c(0, 1), name = "Adj.R²"
    ) +
    labs(
      title = sprintf("Final Models [%s]", criteria_label),
      x = "Band", y = ""
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold"),
      panel.grid = element_blank()
    )

  ggsave(
    file.path(output_dir, sprintf("model_summary_%s.png", criteria_name)),
    p, width = 10, height = 5, dpi = 200
  )
}

plot_model_fits <- function(df, results, leaf_type, output_dir,
                            criteria_name, criteria_label) {
  color <- if (leaf_type == "broadleaf") CB_ROSE
  else if (leaf_type == "needleleaf")   CB_INDIGO
  else                                   CB_PURPLE

  plot_rows <- list()

  for (response in RESPONSE_VARS) {
    res <- results[[response]]
    if (is.null(res) || length(res$final_preds) == 0) next
    resp_short <- gsub("attenuation_|_db", "", response)

    for (pred in res$final_preds) {
      vars_needed <- c("numeric_id", response, pred)
      df_model <- na.omit(df[, ..vars_needed])
      if (nrow(df_model) < 3) next

      r_val <- cor(df_model[[pred]], df_model[[response]])
      plot_rows[[length(plot_rows) + 1]] <- list(
        response = resp_short,
        predictor = pred,
        df_plot = data.frame(
          tree_id = df_model$numeric_id,
          x = df_model[[pred]],
          y = df_model[[response]]
        ),
        r = r_val
      )
    }
  }

  if (length(plot_rows) == 0) return()

  plot_df <- do.call(rbind, lapply(plot_rows, function(pr) {
    data.frame(
      response = pr$response,
      predictor = pr$predictor,
      tree_id = pr$df_plot$tree_id,
      x = pr$df_plot$x,
      y = pr$df_plot$y,
      r = pr$r,
      stringsAsFactors = FALSE
    )
  }))

  plot_df$response <- factor(
    plot_df$response,
    levels = c("low", "mid", "high", "overall"),
    labels = c("Low", "Mid", "High", "Overall")
  )
  plot_df$pred_label <- sapply(
    plot_df$predictor,
    function(p) if (p %in% names(VAR_LABELS)) VAR_LABELS[p] else p
  )

  p <- ggplot(plot_df, aes(x = x, y = y)) +
    geom_smooth(method = "lm", se = FALSE, color = color, linewidth = 1) +
    geom_point(color = color, size = 3, alpha = 0.8) +
    facet_grid(response ~ pred_label, scales = "free") +
    labs(
      title = sprintf(
        "Model Fits: %s [%s]",
        tools::toTitleCase(leaf_type), criteria_label
      ),
      x = "Predictor", y = "Attenuation (dB)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold"),
      strip.text = element_text(face = "bold")
    )

  if (USE_REPEL) {
    p <- p + ggrepel::geom_text_repel(
      aes(label = tree_id),
      size = 2.5, color = "gray30", max.overlaps = 15
    )
  } else {
    p <- p + geom_text(
      aes(label = tree_id),
      size = 2.5, color = "gray30", hjust = -0.2, vjust = -0.5
    )
  }

  n_cols <- length(unique(plot_df$pred_label))
  n_rows <- length(unique(plot_df$response))

  ggsave(
    file.path(output_dir, sprintf("model_fits_%s_%s.png", leaf_type, criteria_name)),
    p,
    width = min(n_cols * 3.5 + 1, 14),
    height = min(n_rows * 3 + 1, 12),
    dpi = 200
  )
}

plot_stepwise_process <- function(results_broad, results_needle, results_mixed,
                                  output_dir, criteria_name, criteria_label) {
  #' Visualize the stepwise selection process for all three leaf types
  message(sprintf(" Creating stepwise process plots for: %s", criteria_name))

  # Determine which metric column to use for plotting
  if (criteria_name == "adj_r2") {
    metric_col   <- "delta_adj_r2"
    metric_label <- "ΔAdj.R²"
    threshold    <- 0.02   # Must improve by 2%
    higher_better <- TRUE
  } else if (criteria_name == "f_test") {
    metric_col   <- "f_pvalue"
    metric_label <- "F-test p-value"
    threshold    <- 0.15   # Relaxed for small n
    higher_better <- FALSE
  } else if (criteria_name == "loocv") {
    metric_col   <- "delta_loocv"
    metric_label <- "ΔLOOCV RMSE"
    threshold    <- 0      # Any improvement
    higher_better <- FALSE
  } else if (criteria_name == "bic") {
    metric_col   <- "delta_bic"
    metric_label <- "ΔBIC"
    threshold    <- 0      # Any improvement
    higher_better <- FALSE
  }

  # Helper function to collect steps from one set of results
  collect_from_results <- function(results, leaf_type_name) {
    all_rows <- list()
    for (resp_name in names(results)) {
      res <- results[[resp_name]]
      if (is.null(res)) next
      resp_short <- gsub("attenuation_|_db", "", resp_name)

      n_steps <- length(res$steps)
      if (n_steps == 0) next

      for (step_idx in seq_along(res$steps)) {
        step <- res$steps[[step_idx]]
        if (is.null(step) || is.null(step$all_candidates)) next

        cands <- step$all_candidates
        selected_pred <- if (!is.null(step$added) && !is.na(step$added)) step$added else NA

        for (i in 1:nrow(cands)) {
          row_data <- data.frame(
            leaf_type    = leaf_type_name,
            response     = resp_short,
            step         = step_idx,
            predictor    = as.character(cands$predictor[i]),
            metric_value = as.numeric(cands[[metric_col]][i]),
            adj_r2       = as.numeric(cands$adj_r2[i]),
            selected     = (!is.na(selected_pred) &&
                             as.character(cands$predictor[i]) == selected_pred),
            stringsAsFactors = FALSE
          )
          all_rows[[length(all_rows) + 1]] <- row_data
        }
      }
    }

    if (length(all_rows) > 0) {
      return(do.call(rbind, all_rows))
    } else {
      return(NULL)
    }
  }

  # Collect data from all three leaf types
  df_broad  <- collect_from_results(results_broad,  "Broadleaf")
  df_needle <- collect_from_results(results_needle, "Needleleaf")
  df_mixed  <- collect_from_results(results_mixed,  "Mixed")

  # Combine
  df_list <- list()
  if (!is.null(df_broad))  df_list[[length(df_list) + 1]] <- df_broad
  if (!is.null(df_needle)) df_list[[length(df_list) + 1]] <- df_needle
  if (!is.null(df_mixed))  df_list[[length(df_list) + 1]] <- df_mixed

  if (length(df_list) == 0) {
    message(" No stepwise data collected!")
    return()
  }

  df_all <- do.call(rbind, df_list)
  message(sprintf(" Total rows collected: %d", nrow(df_all)))

  # Pretty labels
  df_all$pred_label <- ifelse(
    df_all$predictor %in% names(VAR_LABELS),
    VAR_LABELS[df_all$predictor],
    df_all$predictor
  )
  df_all$response <- factor(
    df_all$response,
    levels = c("low", "mid", "high", "overall"),
    labels = c("Low Freq", "Mid Freq", "High Freq", "Overall")
  )
  df_all$step_label <- paste("Step", df_all$step)

  # ===== Create plot for each leaf type =====
  for (lt in c("Broadleaf", "Needleleaf", "Mixed")) {
    df_lt <- df_all[df_all$leaf_type == lt, ]
    if (nrow(df_lt) == 0) {
      message(sprintf(" No data for %s", lt))
      next
    }

    # Color for this leaf type: Rose(broad), Indigo(needle), Purple(mixed)
    sel_color <- if (lt == "Broadleaf") CB_ROSE
    else if (lt == "Needleleaf")       CB_INDIGO
    else                               CB_PURPLE

    # Determine the maximum number of steps across all responses
    max_steps <- max(df_lt$step)
    responses <- levels(df_lt$response)

    tryCatch({
      library(patchwork)

      # Build the combined plot
      all_rows <- list()
      for (resp in responses) {
        df_resp <- df_lt[df_lt$response == resp, ]

        # define a global predictor order for this response so all steps align
        if (higher_better) {
          df_resp_ord <- df_resp[order(-df_resp$metric_value), ]
        } else {
          df_resp_ord <- df_resp[order(df_resp$metric_value), ]
        }
        global_pred_order <- unique(df_resp_ord$pred_label)

        existing_steps <- sort(unique(df_resp$step))
        step_plots <- list()

        # Create plots for all steps 1 to max_steps
        for (st in 1:max_steps) {
          if (st %in% existing_steps) {
            # We have data for this step
            df_step <- df_resp[df_resp$step == st, ]

            # Order predictors within this step to follow the global order
            df_step$pred_label <- factor(
              df_step$pred_label,
              levels = rev(global_pred_order)  # bottom = "best"
            )

            # Create color vector for axis labels
            axis_colors <- ifelse(df_step$selected, "red", "gray30")
            names(axis_colors) <- levels(df_step$pred_label)

            # Create fontface vector
            axis_face <- ifelse(df_step$selected, "bold", "plain")
            names(axis_face) <- levels(df_step$pred_label)

            # Format values for display
            if (criteria_name == "f_test") {
              df_step$label_text <- sprintf("%.3f", df_step$metric_value)
            } else if (criteria_name == "bic") {
              df_step$label_text <- sprintf("%+.0f", df_step$metric_value)
            } else {
              df_step$label_text <- sprintf("%+.2f", df_step$metric_value)
            }

            # Determine hjust based on criterion direction
            if (higher_better) {
              hjust_vec <- ifelse(df_step$metric_value > 0, -0.1, 1.1)
            } else {
              hjust_vec <- ifelse(df_step$metric_value < 0, -0.1, 1.1)
            }

            p_step <- ggplot(df_step, aes(x = pred_label, y = metric_value)) +
              geom_hline(
                yintercept = threshold,
                linetype = "dashed",
                color = "forestgreen", linewidth = 0.6
              ) +
              geom_hline(
                yintercept = 0,
                color = "gray50", linewidth = 0.4
              ) +
              geom_col(aes(fill = selected), width = 0.7) +
              geom_text(
                aes(label = label_text),
                hjust = hjust_vec,
                size = 2.2, color = "gray30"
              ) +
              scale_fill_manual(
                values = c("FALSE" = "gray70", "TRUE" = sel_color),
                guide = "none"
              ) +
              coord_flip() +
              labs(
                x = "", y = metric_label,
                title = sprintf("Step %d", st)
              ) +
              theme_minimal() +
              theme(
                plot.title = element_text(size = 9, hjust = 0.5),
                axis.text.y = element_text(
                  size = 8,
                  color = axis_colors[levels(df_step$pred_label)],
                  face  = axis_face[levels(df_step$pred_label)]
                ),
                axis.text.x   = element_text(size = 7),
                axis.title.x  = element_text(size = 8),
                panel.grid.minor = element_blank(),
                panel.grid.major.y = element_blank(),
                plot.margin = margin(2, 5, 2, 2)
              )
          } else {
            # No data for this step - create empty white placeholder
            p_step <- ggplot() +
              theme_void() +
              labs(title = sprintf("Step %d", st)) +
              theme(
                plot.title = element_text(size = 9, hjust = 0.5, color = "gray70"),
                plot.background = element_rect(fill = "white", color = NA)
              )
          }

          step_plots[[length(step_plots) + 1]] <- p_step
        }

        # Add response label to first plot
        step_plots[[1]] <- step_plots[[1]] +
          labs(tag = resp) +
          theme(
            plot.tag = element_text(face = "bold", size = 10),
            plot.tag.position = "left"
          )

        # Combine horizontally
        row_plot <- wrap_plots(step_plots, nrow = 1)
        all_rows[[length(all_rows) + 1]] <- row_plot
      }

      # Combine all rows vertically
      final_plot <- wrap_plots(all_rows, ncol = 1) +
        plot_annotation(
          title = sprintf("Forward Stepwise Selection: %s Trees [%s]", lt, criteria_label),
          subtitle = "Green line = selection threshold | Red labels = selected predictor",
          theme = theme(
            plot.title = element_text(face = "bold", size = 14),
            plot.subtitle = element_text(size = 10, color = "gray40")
          )
        )

      filename <- sprintf("stepwise_%s_%s.png", tolower(lt), criteria_name)
      ggsave(
        file.path(output_dir, filename),
        final_plot,
        width = 16, height = 14, dpi = 200
      )
      message(sprintf(" Saved: %s", filename))
    }, error = function(e) {
      message(sprintf(" Error creating %s plot: %s", lt, e$message))
    })
  }
}

# ============================================================
# MAIN
# ============================================================

run_analysis <- function(data_path, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  log_path <- file.path(output_dir, "stepwise_log.txt")
  log_file <- file(log_path, "w")

  message("Loading data...")
  df <- fread(data_path)

  df_broad  <- df[leaf_type == "broadleaf"]
  df_needle <- df[leaf_type == "needleleaf"]
  df_mixed  <- df

  message(sprintf(
    " Broadleaf: %d, Needleleaf: %d, Mixed: %d",
    nrow(df_broad), nrow(df_needle), nrow(df_mixed)
  ))

  avail_preds <- PREDICTORS[PREDICTORS %in% names(df)]

  # Collinearity
  analyze_collinearity(df_broad,  avail_preds, output_dir, "Broadleaf")
  analyze_collinearity(df_needle, avail_preds, output_dir, "Needleleaf")
  analyze_collinearity(df_mixed,  avail_preds, output_dir, "Mixed")

  all_results  <- list()
  summary_rows <- list()

  for (crit_name in names(CRITERIA_LIST)) {
    crit <- CRITERIA_LIST[[crit_name]]
    message(sprintf("\n=== Criterion: %s ===", crit$label))

    results_broad  <- list()
    results_needle <- list()
    results_mixed  <- list()

    for (response in RESPONSE_VARS) {
      resp_short <- gsub("attenuation_|_db", "", response)

      results_broad[[response]] <- forward_stepwise(
        df_broad, response, avail_preds,
        max_preds = 3, criterion = crit$name, log_file = log_file
      )
      results_needle[[response]] <- forward_stepwise(
        df_needle, response, avail_preds,
        max_preds = 3, criterion = crit$name, log_file = log_file
      )
      results_mixed[[response]] <- forward_stepwise(
        df_mixed, response, avail_preds,
        max_preds = 3, criterion = crit$name, log_file = log_file
      )

      # Summary rows
      for (lt in list(
        list("broadleaf", results_broad),
        list("needleleaf", results_needle),
        list("mixed",     results_mixed)
      )) {
        res <- lt[[2]][[response]]
        preds_str <- if (is.null(res) || length(res$final_preds) == 0) {
          "(null)"
        } else {
          paste(res$final_preds, collapse = " + ")
        }

        adj_r2 <- if (is.null(res) || length(res$final_preds) == 0) {
          0
        } else {
          summary(res$final_model)$adj.r.squared
        }

        n_obs <- if (is.null(res)) NA else res$n

        summary_rows[[length(summary_rows) + 1]] <- data.frame(
          criterion = crit_name,
          leaf_type = lt[[1]],
          response  = resp_short,
          predictors = preds_str,
          adj_r2    = adj_r2,
          n         = n_obs,
          stringsAsFactors = FALSE
        )
      }
    }

    all_results[[crit_name]] <- list(
      broadleaf  = results_broad,
      needleleaf = results_needle,
      mixed      = results_mixed
    )

    create_summary_heatmap(
      results_broad, results_needle, results_mixed,
      output_dir, crit$name, crit$label
    )
    plot_stepwise_process(
      results_broad, results_needle, results_mixed,
      output_dir, crit$name, crit$label
    )
    plot_model_fits(df_broad,  results_broad,  "broadleaf",  output_dir, crit$name, crit$label)
    plot_model_fits(df_needle, results_needle, "needleleaf", output_dir, crit$name, crit$label)
    plot_model_fits(df_mixed,  results_mixed,  "mixed",      output_dir, crit$name, crit$label)
  }

  close(log_file)

  summary_df <- do.call(rbind, summary_rows)
  fwrite(summary_df, file.path(output_dir, "stepwise_summary.csv"))

  message("\n=== COMPLETE ===")
  message(sprintf("Outputs: %s", output_dir))

  return(all_results)
}

# CLI
if (!interactive()) {
  option_list <- list(
    make_option(c("-d", "--data"), type = "character", default = NULL,
                help = "Path to merged_data.csv"),
    make_option(c("-o", "--output"), type = "character",
                default = DEFAULT_OUTPUT_DIR,
                help = "Output directory")
  )

  opt <- parse_args(OptionParser(option_list = option_list))
  if (is.null(opt$data)) stop("Please provide --data", call. = FALSE)

  run_analysis(opt$data, opt$output)
}
