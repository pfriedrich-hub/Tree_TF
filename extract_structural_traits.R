#############################################
# STRUCTURAL TRAIT EXTRACTION FROM TLS DATA
# For Tree Sound Absorption Analysis
#############################################
#
# Usage:
#   Rscript extract_structural_traits.R --data-dir /path/to/las --output-dir ./output
#
# Or interactively in RStudio

# ============================================================
# PACKAGES
# ============================================================
suppressPackageStartupMessages({
  library(lidR)
  library(VoxR)
  library(dplyr)
  library(geometry)
  library(data.table)
  library(optparse)
  library(ggplot2)
})

# ============================================================
# CONFIGURATION
# ============================================================

# Default paths - adjust to your system
DEFAULT_TLS_DIR <- "/home/tigris/DATA/Arboretum"
DEFAULT_OUTPUT_DIR <- "output"

# Note: 313, 344, 353 excluded (linear sweeps)
TARGET_IDS <- c(
  227, 232, 247, 257, 270, 274, 277, 281, 298,
  327, 332, 333, 342, 467, 499, 502, 518
)

SPECIES_MAP <- list(
  "227" = "Lar_dec", "232" = "Pru_avi", "247" = "Pop_tre",
  "257" = "Pse_men", "270" = "Aln_glu", "274" = "Pop_tre",
  "277" = "Pru_avi", "281" = "Lar_dec", "298" = "Abi_gra",
  "327" = "Sal_cap", "332" = "Pin_nig",
  "333" = "Pin_nig", "342" = "Pse_men",
  "467" = "Til_tom", "499" = "Til_tom",
  "502" = "Ced_deo", "518" = "Til_tom"
)

# Measurement angles (degrees from North, clockwise)
MEASUREMENT_ANGLES <- list(
  "227" = 320, "232" = 230, "247" = 318, "257" = 273,
  "270" = 240, "274" = 250, "277" = 120, "281" = 190,
  "298" = 336, "313" = 20, "327" = 170, "332" = 55,
  "333" = 325, "342" = 80, "344" = 261, "353" = 255,
  "467" = 270, "499" = 40, "502" = 46, "518" = 0
)

# Processing parameters
VOXEL_SIZE <- 0.05
IVF_RES <- 0.02
IVF_N <- 6
CIRCLE_RADIUS <- 3.0

# Colorblind-friendly palette
# Color scheme: red(broad) + blue(needle) = purple(mixed)
CB_ROSE <- "#CC6677"       # Broadleaf (reddish)
CB_INDIGO <- "#332288"     # Needleleaf (bluish)
CB_PURPLE <- "#AA4499"     # Mixed analysis (purple)
CB_FOREST <- "#117733"     # Tree/vegetation (darker green)
CB_GOLD <- "#DDCC77"       # Highlights/markers (yellow-ish, visible on green)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

find_las_file <- function(id, data_dir) {
  id_str <- as.character(id)
  species <- SPECIES_MAP[[id_str]]
  
  if (is.null(species)) {
    message(sprintf("  No species mapping for ID %s", id_str))
    return(NA)
  }
  
  patterns <- c(
    paste0("^", species, "_", id_str, "\\.las$"),
    paste0("^", species, "_", id_str, ".*\\.las$"),
    paste0("^", id_str, "\\.las$"),
    paste0(".*_", id_str, "\\.las$")
  )
  
  for (pattern in patterns) {
    files <- list.files(data_dir, pattern = pattern, full.names = TRUE, 
                        ignore.case = TRUE)
    if (length(files) > 0) return(files[1])
  }
  
  return(NA)
}


preprocess_las <- function(las) {
  las <- filter_duplicates(las)
  las <- classify_noise(las, ivf(res = IVF_RES, n = IVF_N))
  las <- filter_poi(las, Classification != LASNOISE)
  
  z0 <- as.numeric(quantile(las$Z, probs = 0.001, na.rm = TRUE))
  z_shifted <- las$Z - z0
  stem_mask <- z_shifted >= 0.8 & z_shifted <= 1.3
  
  if (sum(stem_mask) > 10) {
    cx <- median(las$X[stem_mask], na.rm = TRUE)
    cy <- median(las$Y[stem_mask], na.rm = TRUE)
  } else {
    cx <- median(las$X, na.rm = TRUE)
    cy <- median(las$Y, na.rm = TRUE)
  }
  
  df <- data.frame(
    X = las$X - cx,
    Y = las$Y - cy,
    Z = z_shifted
  )
  
  z_max <- as.numeric(quantile(df$Z, probs = 0.999, na.rm = TRUE))
  df <- df[df$Z >= 0 & df$Z <= z_max, ]
  
  las_new <- LAS(df)
  return(las_new)
}


# ============================================================
# STRUCTURAL TRAIT COMPUTATION
# ============================================================

compute_structural_traits <- function(las, id, voxel_size = VOXEL_SIZE) {
  pts <- data.frame(X = las$X, Y = las$Y, Z = las$Z)
  id_str <- as.character(id)
  
  # Height
  height_m <- as.numeric(quantile(pts$Z, 0.99) - quantile(pts$Z, 0.01))
  
  # Crown Volume: 3d convex hull
  crown_volume_m3 <- tryCatch({
    hull <- convhulln(as.matrix(pts), options = "FA")
    hull$vol
  }, error = function(e) NA)
  
  # Voxelization
  vx <- VoxR::vox(pts[, c("X", "Y", "Z")], res = voxel_size)
  n_voxels <- nrow(vx)
  pts_per_voxel <- nrow(pts) / n_voxels
  
  # Volumetric Fill Fraction
  vol_fill_fraction <- tryCatch({
    # Create grid covering bounding box
    vx_coords <- as.matrix(vx)
    min_xyz <- apply(vx_coords, 2, min)
    max_xyz <- apply(vx_coords, 2, max)
    
    x_seq <- seq(min_xyz[1], max_xyz[1], by = voxel_size)
    y_seq <- seq(min_xyz[2], max_xyz[2], by = voxel_size)
    z_seq <- seq(min_xyz[3], max_xyz[3], by = voxel_size)
    grid <- expand.grid(X = x_seq, Y = y_seq, Z = z_seq)

    # Count voxels inside convex hull
    hull <- convhulln(as.matrix(pts))
    inside <- inhulln(hull, as.matrix(grid))
    n_hull_voxels <- sum(inside)

    # Ratio: occupied / possible
    n_voxels / n_hull_voxels
  }, error = function(e) NA)
  
  # Densest Slice Height
  z_bins <- seq(0, max(vx$z) + voxel_size, by = 0.1)
  bin_idx <- findInterval(vx$z, z_bins, rightmost.closed = TRUE)
  slice_counts <- tabulate(bin_idx, nbins = length(z_bins))
  densest_bin <- which.max(slice_counts)
  densest_slice_height_m <- z_bins[densest_bin] + 0.05
  
  # Leaf Voxel Density at Measurement Height
  slice_mask <- vx$z >= (densest_slice_height_m - 0.05) & 
                vx$z < (densest_slice_height_m + 0.05)
  leaf_voxel_density <- sum(slice_mask)
  
  slice_pts <- vx[slice_mask, ]
  
  # Gap Fraction with projection
  gap_result <- tryCatch({
    compute_gap_fraction_with_projection(vx, slice_pts, id_str, voxel_size)
  }, error = function(e) {
    message(sprintf("  Warning: Gap fraction failed: %s", e$message))
    list(gap_fraction = NA, hit_point = c(0,0,0), direction = c(0,1,0), clip_bottom = 0)
  })
  
  # ENL (Effective Number of Layers) - Quantifies vertical canopy layering
  # Measures how many effectively distinct height layers exist (1 = single layer, N = N even layers)

  enl_bins <- seq(0, max(vx$z) + 0.5, by = 0.5)              # Create 0.5m height bins from ground to tree top
  enl_bin_idx <- findInterval(vx$z, enl_bins, rightmost.closed = TRUE)  # Assign each point to its height bin
  enl_counts <- tabulate(enl_bin_idx, nbins = length(enl_bins))         # Count points per height bin
  enl_counts <- enl_counts[enl_counts > 0]                              # Remove empty bins
  enl_props <- enl_counts / sum(enl_counts)                             # Calculate proportion of points per occupied bin
  ENL <- 1 / sum(enl_props^2)                                           # Simpson diversity index: 1/sum(p_i^2)

  # FHD (Foliage Height Diversity) - Shannon entropy
  FHD <- -sum(enl_props * log2(enl_props + 1e-12))
  
  # LAD (Leaf Area Density) - from gap fraction using Beer-Lambert
  # ... tried to adapt from ligt to sound, questionable
  LAD <- tryCatch({
    gf <- gap_result$gap_fraction
    if (is.na(gf) || gf <= 0 || gf >= 1) {
      NA
    } else {
      crown_depth <- max(vx$z) - min(vx$z)
      if (crown_depth <= 0) crown_depth <- 1
      LAI <- -log(gf) / 0.5  # Assuming extinction coefficient k = 0.5
      LAI / crown_depth
    }
  }, error = function(e) NA)
  
  return(list(
    height_m = height_m,
    crown_volume_m3 = crown_volume_m3,
    pts_per_voxel = pts_per_voxel,
    vol_fill_fraction = vol_fill_fraction,
    densest_slice_height_m = densest_slice_height_m,
    gap_fraction = gap_result$gap_fraction,
    leaf_voxel_density = leaf_voxel_density,
    ENL = ENL,
    FHD = FHD,
    LAD = LAD,
    vx = vx,
    hit_point = gap_result$hit_point,
    direction = gap_result$direction,
    clip_bottom = gap_result$clip_bottom,
    proj_dt = gap_result$proj_dt
  ))
}


compute_gap_fraction_with_projection <- function(vx, slice_pts, id_str, voxel_size) {
  # Get measurement angle
  angle_deg <- MEASUREMENT_ANGLES[[id_str]]
  if (is.null(angle_deg)) angle_deg <- 0
  
  cx <- median(slice_pts$x)
  cy <- median(slice_pts$y)
  cz <- median(slice_pts$z)
  center3 <- c(cx, cy, cz)
  
  theta <- angle_deg * pi / 180

  # Direction vector (horizontal plane, from speaker into tree)
  dir <- c(sin(theta), cos(theta), 0)
  
  # Walk along the sound array to find first intersection with points
  # -> find first voxel hit (hit_point)
  step <- voxel_size / 2
  start_point <- center3 - dir * (5 * CIRCLE_RADIUS)
  hit_point <- center3
  
  for (k in 1:2000) {
    p <- start_point + dir * (k * step)
    tol <- voxel_size / 2
    hit_idx <- which(
      abs(vx$x - p[1]) < tol &
      abs(vx$y - p[2]) < tol &
      abs(vx$z - p[3]) < tol
    )
    if (length(hit_idx) > 0) {
      hit_point <- p
      break
    }
  }
  
  # Project all voxels onto plane perpendicular to sound path
  lateral <- c(cos(theta), -sin(theta), 0)
  vertical <- c(0, 0, 1)
  
  vx_mat <- as.matrix(vx[, c("x", "y", "z")])
  proj <- t(apply(vx_mat, 1, function(pt) {
    v <- pt - hit_point
    c(sum(v * lateral), sum(v * vertical))
  }))
  proj_dt <- data.table(u = proj[,1], w = proj[,2])
  
  w_min <- min(proj_dt$w)
  clip_bottom <- max(w_min, -CIRCLE_RADIUS)
  
  # Count voxels inside 3m radius circle above clip_bottom
  d <- sqrt(proj_dt$u^2 + proj_dt$w^2)
  inside_mask <- (d <= CIRCLE_RADIUS) & (proj_dt$w >= clip_bottom)
  
  if (sum(inside_mask) == 0) {
    return(list(gap_fraction = 1.0, hit_point = hit_point, direction = dir, 
                clip_bottom = clip_bottom, proj_dt = proj_dt))
  }
  
  proj_inside <- proj_dt[inside_mask]
  gx <- floor((proj_inside$u + CIRCLE_RADIUS) / voxel_size)
  gy <- floor((proj_inside$w + CIRCLE_RADIUS) / voxel_size)
  occupied <- length(unique(paste(gx, gy, sep = "_")))
  
  u_seq <- seq(-CIRCLE_RADIUS, CIRCLE_RADIUS, by = voxel_size)
  w_seq <- seq(clip_bottom, CIRCLE_RADIUS, by = voxel_size)
  grid <- expand.grid(u = u_seq, w = w_seq)
  inside_grid <- subset(grid, u^2 + w^2 <= CIRCLE_RADIUS^2)
  n_total <- nrow(inside_grid)
  
  # Gap fraction = 1 - (occupied / total)
  gap_fraction <- 1 - (occupied / n_total)
  
  return(list(
    gap_fraction = gap_fraction, 
    hit_point = hit_point, 
    direction = dir,
    clip_bottom = clip_bottom,
    proj_dt = proj_dt
  ))
}


# ============================================================
# VISUALIZATION
# ============================================================

plot_topdown_view <- function(vx, pts, hit_point, dir, id, species, output_dir) {
  pts_dt <- as.data.table(pts)
  
  pts_dt[, ix := floor(X / VOXEL_SIZE)]
  pts_dt[, iy := floor(Y / VOXEL_SIZE)]
  occ <- pts_dt[, .(count = .N), by = .(ix, iy)]
  occ[, xcenter := (ix + 0.5) * VOXEL_SIZE]
  occ[, ycenter := (iy + 0.5) * VOXEL_SIZE]
  occ[, log_count := log10(count + 1)]
  
  tree_center <- c(0, 0)
  arrow_length <- sqrt(sum((tree_center - hit_point[1:2])^2))
  arrow_end <- hit_point[1:2] + dir[1:2] * arrow_length
  
  # Darker green palette for forest canopy
  green_palette <- c("#f7fcf5", "#a1d99b", "#41ab5d", "#238b45", "#005a32")
  
  p <- ggplot() +
    geom_tile(data = occ, aes(x = xcenter, y = ycenter, fill = log_count)) +
    scale_fill_gradientn(colors = green_palette, name = "Density\n(log₁₀)", na.value = "white") +
    geom_segment(
      aes(x = hit_point[1], y = hit_point[2], xend = arrow_end[1], yend = arrow_end[2]),
      arrow = arrow(length = unit(0.25, "cm"), type = "closed"),
      color = CB_GOLD, linewidth = 1.0
    ) +
    geom_point(aes(x = hit_point[1], y = hit_point[2]), color = CB_GOLD, size = 3, shape = 16) +
    geom_point(aes(x = 0, y = 0), color = "#444444", size = 2, shape = 3) +
    coord_equal() +
    labs(title = sprintf("%s (%s)", species, id), subtitle = "Top-down view · Arrow = sound path",
         x = "X (m)", y = "Y (m)") +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 13),
          plot.subtitle = element_text(color = "grey40", size = 10),
          panel.grid.minor = element_blank(),
          plot.background = element_rect(fill = "white", color = NA))
  
  ggsave(file.path(output_dir, sprintf("%s_%s_topdown.png", species, id)), p, width = 7, height = 6, dpi = 200)
}


plot_projection_view <- function(proj_dt, clip_bottom, id, species, output_dir) {
  d <- sqrt(proj_dt$u^2 + proj_dt$w^2)
  
  plot_df <- data.frame(u = proj_dt$u, w = proj_dt$w, inside = (d <= CIRCLE_RADIUS) & (proj_dt$w >= clip_bottom))
  
  ang <- seq(0, 2*pi, length.out = 200)
  circle_df <- data.frame(u = CIRCLE_RADIUS * cos(ang), w = CIRCLE_RADIUS * sin(ang))
  circle_df <- subset(circle_df, w >= clip_bottom)
  
  p <- ggplot() +
    geom_point(data = subset(plot_df, !inside), aes(x = u, y = w), color = "grey80", size = 0.6, alpha = 0.4) +
    geom_point(data = subset(plot_df, inside), aes(x = u, y = w), color = CB_FOREST, size = 0.8, alpha = 0.7) +
    geom_path(data = circle_df, aes(u, w), color = "#444444", linewidth = 0.8) +
    geom_segment(aes(x = -CIRCLE_RADIUS, y = clip_bottom, xend = CIRCLE_RADIUS, yend = clip_bottom),
                 color = CB_GOLD, linetype = "dashed", linewidth = 0.8) +
    geom_point(aes(x = 0, y = 0), color = CB_GOLD, size = 3, shape = 16) +
    coord_equal() +
    labs(title = sprintf("%s (%s)", species, id),
         subtitle = sprintf("Speaker view · Ground clip at %.1f m", clip_bottom),
         x = "Lateral (m)", y = "Vertical (m)") +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 13),
          plot.subtitle = element_text(color = "grey40", size = 10),
          legend.position = "none",
          plot.background = element_rect(fill = "white", color = NA))
  
  ggsave(file.path(output_dir, sprintf("%s_%s_projection.png", species, id)), p, width = 6, height = 6, dpi = 200)
}


# ============================================================
# MAIN PIPELINE
# ============================================================

process_single_tree <- function(id, data_dir, output_dir, make_plots = TRUE) {
  id_str <- as.character(id)
  message(sprintf("Processing tree %s...", id_str))
  
  las_file <- find_las_file(id, data_dir)
  if (is.na(las_file)) {
    message(sprintf("  Warning: No LAS file found for ID %s", id_str))
    return(NULL)
  }
  
  message(sprintf("  Loading: %s", basename(las_file)))
  
  las <- readLAS(las_file, select = "xyz")
  if (is.empty(las)) {
    message("  Warning: Empty point cloud")
    return(NULL)
  }
  
  las <- preprocess_las(las)
  pts <- data.frame(X = las$X, Y = las$Y, Z = las$Z)
  
  traits <- compute_structural_traits(las, id, VOXEL_SIZE)
  traits$numeric_id <- as.integer(id)
  traits$species <- SPECIES_MAP[[id_str]]
  
  if (make_plots && !is.na(traits$gap_fraction)) {
    plot_dir <- file.path(output_dir, "figures", "structural")
    dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
    
    tryCatch({
      plot_topdown_view(traits$vx, pts, traits$hit_point, traits$direction, id_str, traits$species, plot_dir)
      plot_projection_view(traits$proj_dt, traits$clip_bottom, id_str, traits$species, plot_dir)
    }, error = function(e) {
      message(sprintf("  Warning: Plotting failed: %s", e$message))
    })
  }
  
  message(sprintf("  Height: %.2f m, Crown vol: %.1f m³, Gap: %.2f, ENL: %.2f", 
                  traits$height_m, traits$crown_volume_m3, 
                  ifelse(is.na(traits$gap_fraction), NA, traits$gap_fraction),
                  ifelse(is.na(traits$ENL), NA, traits$ENL)))
  
  return(list(
    numeric_id = traits$numeric_id,
    species = traits$species,
    height_m = traits$height_m,
    crown_volume_m3 = traits$crown_volume_m3,
    pts_per_voxel = traits$pts_per_voxel,
    vol_fill_fraction = traits$vol_fill_fraction,
    densest_slice_height_m = traits$densest_slice_height_m,
    gap_fraction = traits$gap_fraction,
    leaf_voxel_density = traits$leaf_voxel_density,
    ENL = traits$ENL,
    FHD = traits$FHD,
    LAD = traits$LAD
  ))
}


run_structural_extraction <- function(data_dir, output_dir, make_plots = TRUE) {
  message("============================================================")
  message("STRUCTURAL TRAIT EXTRACTION")
  message("============================================================")
  message(sprintf("Data directory: %s", data_dir))
  message(sprintf("Output directory: %s", output_dir))
  message(sprintf("Processing %d trees", length(TARGET_IDS)))
  message("")
  
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  results <- list()
  for (id in TARGET_IDS) {
    traits <- process_single_tree(id, data_dir, output_dir, make_plots)
    if (!is.null(traits)) {
      results[[length(results) + 1]] <- traits
    }
  }
  
  if (length(results) == 0) {
    stop("No trees processed successfully!")
  }
  
  df <- rbindlist(results, fill = TRUE)
  
  # Define output column order
  col_order <- c("numeric_id", "species", "height_m", "crown_volume_m3", "pts_per_voxel", 
                 "vol_fill_fraction", "densest_slice_height_m", "gap_fraction", 
                 "leaf_voxel_density", "ENL", "FHD", "LAD")
  col_order <- col_order[col_order %in% names(df)]
  df <- df[, ..col_order]
  
  output_path <- file.path(output_dir, "csv", "tree_structural_traits.csv")
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  fwrite(df, output_path)
  
  message("")
  message(sprintf("Results saved to: %s", output_path))
  message(sprintf("Processed %d trees successfully", nrow(df)))
  
  if (make_plots) {
    message(sprintf("Plots saved to: %s", file.path(output_dir, "figures", "structural")))
  }
  
  return(df)
}


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if (!interactive()) {
  option_list <- list(
    make_option(c("-d", "--data-dir"), type = "character", default = NULL,
                help = "Directory containing LAS files"),
    make_option(c("-o", "--output-dir"), type = "character", default = DEFAULT_OUTPUT_DIR,
                help = sprintf("Output directory [default: %s]", DEFAULT_OUTPUT_DIR)),
    make_option(c("--no-plots"), action = "store_true", default = FALSE,
                help = "Skip plot generation")
  )
  
  opt_parser <- OptionParser(option_list = option_list)
  opt <- parse_args(opt_parser)
  
  if (is.null(opt$`data-dir`)) {
    opt$`data-dir` <- DEFAULT_TLS_DIR
  }
  
  run_structural_extraction(
    data_dir = opt$`data-dir`,
    output_dir = opt$`output-dir`,
    make_plots = !opt$`no-plots`
  )
}
