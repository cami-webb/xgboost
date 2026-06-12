# XGBoost forecast — tree growth and mortality
# Called by run_forecast.R or directly: Rscript R/01_fit_forecast.R

.libPaths(c("/projectnb/dietzelab/cwebb16/R_libs", .libPaths()))

library(aws.s3)
library(yaml)
library(readr)
library(dplyr)
library(xgboost)

cfg  <- read_yaml("config.yml")
tcfg <- cfg$targets[[cfg$target]]
set.seed(cfg$model$random_seed)

# ── S3 setup ───────────────────────────────────────────────────────────────────
Sys.setenv(
  AWS_ACCESS_KEY_ID     = Sys.getenv("OSN_KEY"),
  AWS_SECRET_ACCESS_KEY = Sys.getenv("OSN_SECRET"),
  AWS_DEFAULT_REGION    = ""
)
base_url <- gsub("https://", "", cfg$s3$endpoint)

# ── Load covariates from S3 ───────────────────────────────────────────────────
message("Reading covariates from S3...")
covariate_url <- paste0(cfg$s3$endpoint, "/", cfg$s3$read_bucket, "/", tcfg$covariate_s3_key)
covs <- readr::read_csv(covariate_url, show_col_types = FALSE) |>
  mutate(log_dbh_init = log(dbh_init)) |>
  filter(dbh_init > 0, is.finite(log_dbh_init))

# Species grouping
common_species <- covs |> count(species, sort = TRUE) |>
  filter(n >= 20) |> pull(species)
covs <- covs |> mutate(species_grp = factor(
  if_else(species %in% common_species, species, "Other"),
  levels = c(sort(common_species), "Other")
))

# XGBoost requires numeric matrix — one-hot encode species_grp
make_matrix <- function(df, features) {
  num_cols <- features[features != "species_grp"]
  num_mat  <- as.matrix(df[, intersect(num_cols, names(df)), drop = FALSE])
  if ("species_grp" %in% features && "species_grp" %in% names(df)) {
    spp_dummies <- model.matrix(~ species_grp - 1, data = df)
    num_mat <- cbind(num_mat, spp_dummies)
  }
  num_mat[is.na(num_mat)] <- 0
  num_mat
}

features <- c("log_dbh_init", tcfg$covariates)

# ── Fit and forecast for each interval ────────────────────────────────────────
all_forecasts <- list()

for (iv in tcfg$intervals) {
  message(sprintf("\n=== Interval %s (%d -> %d) ===", iv$name, iv$start_year, iv$end_year))

  train_g <- covs |>
    filter(!is.na(growth_cm_yr), is.na(dead_0621) | dead_0621 == 0L) |>
    filter(!is.na(log_dbh_init))

  train_m <- covs |>
    filter(!is.na(dead_0621)) |>
    filter(!is.na(log_dbh_init))

  # --- Growth XGBoost ---
  message("  Fitting growth XGBoost (n=", nrow(train_g), ")")
  X_g <- make_matrix(train_g, features)
  y_g <- train_g$growth_cm_yr
  dtrain_g <- xgb.DMatrix(data = X_g, label = y_g)

  xgb_growth <- xgb.train(
    params = list(
      objective        = "reg:squarederror",
      max_depth        = cfg$model$max_depth,
      eta              = cfg$model$eta,
      subsample        = cfg$model$subsample,
      colsample_bytree = cfg$model$colsample_bytree,
      seed             = cfg$model$random_seed
    ),
    data      = dtrain_g,
    nrounds   = cfg$model$nrounds,
    verbose   = 0
  )

  # --- Mortality XGBoost ---
  message("  Fitting mortality XGBoost (n=", nrow(train_m), ")")
  X_m <- make_matrix(train_m, features)
  y_m <- as.numeric(train_m$dead_0621)
  dtrain_m <- xgb.DMatrix(data = X_m, label = y_m)

  xgb_mort <- xgb.train(
    params = list(
      objective        = "binary:logistic",
      max_depth        = cfg$model$max_depth,
      eta              = cfg$model$eta,
      subsample        = cfg$model$subsample,
      colsample_bytree = cfg$model$colsample_bytree,
      seed             = cfg$model$random_seed
    ),
    data      = dtrain_m,
    nrounds   = cfg$model$nrounds,
    verbose   = 0
  )

  # --- Predictions ---
  pred_growth <- predict(xgb_growth, newdata = dtrain_g)
  pred_mort   <- predict(xgb_mort,   newdata = dtrain_m)

  # --- Build EFI-format submission ---
  target_datetime <- as.Date(paste0(iv$end_year,   "-01-01"))
  ref_datetime    <- as.Date(paste0(iv$start_year, "-01-01"))

  forecast_growth <- data.frame(
    model_id           = tcfg$model_id,
    datetime           = target_datetime,
    reference_datetime = ref_datetime,
    site_id            = train_g$tree_id,
    family             = "normal",
    parameter          = "mu",
    variable           = "growth_cm_yr",
    prediction         = pred_growth,
    project_id         = cfg$project_id,
    duration           = tcfg$duration,
    interval           = iv$name
  )

  forecast_mort <- data.frame(
    model_id           = tcfg$model_id,
    datetime           = target_datetime,
    reference_datetime = ref_datetime,
    site_id            = train_m$tree_id,
    family             = "bernoulli",
    parameter          = "prob",
    variable           = "dead_0621",
    prediction         = pred_mort,
    project_id         = cfg$project_id,
    duration           = tcfg$duration,
    interval           = iv$name
  )

  all_forecasts[[iv$name]] <- bind_rows(forecast_growth, forecast_mort)
  message(sprintf("  Done interval %s", iv$name))
}

# ── Combine and write to S3 ───────────────────────────────────────────────────
combined      <- bind_rows(all_forecasts)
forecast_file <- paste0("tree-", Sys.Date(), "-xgboost.csv")

write_csv(combined, forecast_file)

s3_key <- paste0(cfg$s3$submissions_path, "/", forecast_file)
message("\nUploading to s3://", cfg$s3$write_bucket, "/", s3_key)

aws.s3::put_object(
  file      = forecast_file,
  object    = s3_key,
  bucket    = cfg$s3$write_bucket,
  base_url  = base_url,
  use_https = TRUE,
  region    = ""
)

unlink(forecast_file)
message("Done. Rows written: ", nrow(combined))
