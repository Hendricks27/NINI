#===============================================================
# PACKAGES
#===============================================================
#install.packages('data.table')
#install.packages('roll')
#install.packages('ranger')
#install.packages('tidyverse')
library(data.table)
library(roll)
library(ranger)
library(tidyverse)
library(e1071)

#===============================================================
# FUNCTIONS
#===============================================================

# Mean absolute error
mae <- function(y, pred) {
  mean(abs(y - pred))
}
# Calculates the coefficient of an AR(1) process z
ar1 <- function(z) {
  cor(z[-length(z)], z[-1])  
}

# Calculates a buntch of statistics from vector x
univariate_stats <- function(x, tag = NULL, p = c(0, 0.05,0.25, 0.75, 0.95,1)) {
  x <- x[!is.na(x)]
  
  out <- c(
    MEAN = mean(x),
    SD  = sd(x),
    setNames(quantile(x, p = p, names = FALSE), paste0("q", p)),
    ar1 = ar1(x),
    kur=kurtosis(x),
    skew=skewness(x))
  
  if (is.null(tag)) 
    return(out)
  
  names(out) <- paste(names(out), tag, sep = "_")
  out
}

# Feature extraction on vector x. Basically calls "univariate_stats" on differently transformed x
create_X <- function(x, rolling_windows = c(10, 100,500,1000,1500)) {
  stats_full <- univariate_stats(x = x, "full", p = c(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100) / 100)
  stats_abs <- univariate_stats(x = abs(x), "abs")
  
  
  # Rolling versions of x
  x_mat <- as.matrix(x, ncol = 1)
  roll_sd_k <- lapply(rolling_windows, function(k) roll_sd(x_mat, width = k))
  
  # Derive stats from rolling versions
  stats_roll_sd <- Map(univariate_stats, roll_sd_k, tag = paste("roll_sd", rolling_windows, sep = "_"))
  
  c(stats_full, stats_abs, unlist(stats_roll_sd))
}

#===============================================================
# CONSTANTS
#===============================================================

# Length of test data sets
n_test <- 150000
# By how much do we shift the time window of n_test rows within earthquake?
stride <- 75000

# Positions of earthquakes (see answers in https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/77390). Used to create contiguous folds for cross-validation and train/validation split
earthquakes <- c(
  5656573,
  50085877,
  104677355,
  138772452,
  187641819,
  218652629,
  245829584,
  307838916,
  338276286,
  375377847,
  419368879,
  461811622,
  495800224,
  528777114,
  585568143,
  621985672) + 1
# Figure out colnames of input as well as number of features
raw <- fread(file.path('/Users/wangrunqiu/Downloads/Kaggle/NINI-ytwang/ytwang/train.csv'),nrows = 150000, data.table = FALSE)
names_input <- names(raw)
names_features <- names(create_X(raw[[1]]))

# If not yet created, make directory to save folds
fold_dir <- file.path("strides", stride)
if (!dir.exists(fold_dir)) {
  dir.create(fold_dir, recursive = TRUE)  
}

for (prep_fold in seq_along(earthquakes)) {# prep_fold <- 1
  cat(prep_fold, "\n")
  
  # Read data between two earthquakes
  raw <- fread(file.path('/Users/wangrunqiu/Downloads/Kaggle/NINI-ytwang/ytwang/train.csv'), 
               nrows = c(earthquakes[1], diff(earthquakes))[prep_fold], 
               skip = c(0, earthquakes)[prep_fold] + 1)
  setnames(raw, names_input)
  
  # How many times do we calculate features for this data chunk?
  n_steps <- (nrow(raw) - n_test) %/% stride
  # Init feature matrix and vector of response
  y <- rep(0,n_steps)
  X <- matrix(NA, nrow = n_steps, ncol = length(names_features), dimnames = list(NULL, names_features))
  
  # Loop through chunk and build up y and X
  pb <- txtProgressBar(0, n_steps, style = 3)
  
  for (i in seq_len(n_steps)) {
    setTxtProgressBar(pb, i)
    from <- 1 + stride * (i - 1)
    to <- n_test + stride * (i - 1)
    X[i, ] <- create_X(raw$acoustic_data[from:to])
    y[i] <- raw$time_to_failure[to]
  }
  
  save(y, X, file = file.path(fold_dir, paste0("fold_", prep_fold, ".RData")))
}

for (i in seq_along(earthquakes)) {
  load(file.path("strides", stride, paste0("fold_", i, ".RData")))
  
  fold <- rep(i, length(y))
  
  if (i == 1) {
    X_mat <- X
    y_vec <- y
    fold_vec <- fold 
  } else {
    X_mat <- rbind(X_mat, X)
    y_vec <- c(y_vec, y)
    fold_vec <- c(fold_vec, fold)
  }
}

form <- reformulate(colnames(X_mat), "label")
fullDF <- data.frame(label = y_vec, X_mat)

#save data frame
fwrite(fullDF, "/Users/wangrunqiu/Downloads/Kaggle/NINI-ytwang/ytwang/full_DF.csv")

# cross-validation
m_fold <- length(earthquakes)
cv <- rep(0,m_fold)
pb <- txtProgressBar(0, m_fold, style = 3)

for (j in seq_along(cv)) { # j <- 1
  setTxtProgressBar(pb, j)
  fit <- ranger(form, fullDF[fold_vec != j, ], seed = 3564 + 54 * j, verbose = 0)
  cv[j] <- mae(fullDF[fold_vec == j, "label"], predict(fit, fullDF[fold_vec == j, ])$predictions)
}

# Resulting score
mean(cv)
weighted.mean(cv, w = tabulate(fold_vec, nbins = m_fold))

# retrain on full data for submission
fit_rf <- ranger(form, fullDF, importance = "impurity", seed = 345)

# Variable importance
par(mar = c(5, 10, 1, 1))
barplot(importance(fit_rf) %>% sort %>% tail(60), horiz = T, las = 1)


# submission
submission <- fread(file.path("/Users/wangrunqiu/Downloads/Kaggle/NINI-ytwang/ytwang/sample_submission.csv"))

# Load each test data and create the feature matrix. Takes 2-3 minutes and can be skipped
# if playing with strides (and not with features)
load_and_prepare <- function(file) {
  seg <- fread(file.path("/Users/wangrunqiu/Downloads/Kaggle/NINI-ytwang/ytwang/test", paste0(file, ".csv")))
  create_X(seg$acoustic_data)
}
all_test <- lapply(submission$seg_id, load_and_prepare)
all_test2 <- do.call(rbind, all_test)
save(all_test2, file = "test_prep.RData")

# save test data
all_test2_new=as.data.frame(all_test2)
names = 
fwrite(all_test2_new, '/Users/wangrunqiu/Downloads/Kaggle/Runqiu Wang/alltest.csv')

# load("test_prep.RData") 
dim(all_test2) # 2624   35

submission$time_to_failure <- predict(fit_rf, data.frame(all_test2))$prediction

head(submission)

# Save
fwrite(submission, paste0("/Users/wangrunqiu/Downloads/Kaggle/NINI-ytwang/ytwang/submission_rf_new_1", stride, ".csv"))
