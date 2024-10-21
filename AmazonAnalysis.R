# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(doParallel)

# Set up parallelization
num_cores <- 4
cl <- makePSOCKcluster(num_cores)

# Read the data
train_dirty <- vroom("train.csv") %>%
  mutate(ACTION = factor(ACTION))
test_dirty <- vroom("test.csv")

# Make a recipe
recipe <- recipe(ACTION ~ ., data = train_dirty) %>%
  step_mutate_at(all_double_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())
prepped_recipe <- prep(recipe)
clean_data <- bake(prepped_recipe, new_data = train_dirty)

# Create folds in the data
folds <- vfold_cv(train_dirty, v = 10, repeats = 1)

#######################
# Logistic Regression #
#######################

# Create a model
logistic_model <- logistic_reg() %>%
  set_engine("glm")

# Create the workflow
logistic_workflow <- workflow() %>%
  add_model(logistic_model) %>%
  add_recipe(recipe)

# Fit and make predictions
logistic_fit <- fit(logistic_workflow, data = train_dirty)
logistic_predictions <- predict(logistic_fit,
                                new_data = test_dirty,
                                type = "prob")$.pred_1

# Write output
logistic_output <- tibble(id = test_dirty$id,
                          Action = logistic_predictions)
vroom_write(logistic_output, "logistic_regression.csv", delim = ",")

#################################
# Penalized Logistic Regression #
#################################

# Create a model
penalized_model <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

# Create the workflow
penalized_workflow <- workflow() %>%
  add_model(penalized_model) %>%
  add_recipe(recipe)

# Set up parallelization
registerDoParallel(cl)

# Tuning
penalized_tuning_grid <- grid_regular(penalty(), mixture(), levels = 5)
penalized_cv_results <- penalized_workflow %>%
  tune_grid(resamples = folds,
            grid = penalized_tuning_grid,
            metrics = metric_set(roc_auc))

stopCluster(cl)

# Get the best tuning parameters
penalized_besttune <- penalized_cv_results %>%
  select_best(metric = "roc_auc")

# Fit and make predictions
penalized_fit <- penalized_workflow %>%
  finalize_workflow(penalized_besttune) %>%
  fit(data = train_dirty)
penalized_predictions <- predict(penalized_fit,
                                 new_data = test_dirty,
                                 type = "prob")$.pred_1

# Write output
penalized_output <- tibble(id = test_dirty$id,
                           Action = penalized_predictions)
vroom_write(penalized_output, "penalized_logistic_regression.csv", delim = ",")

#######################
# K-Nearest Neighbors #
#######################

# Create a model
knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

# Create the workflow
knn_workflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(recipe)

# Set up parallelization
registerDoParallel(cl)

# Tuning
knn_tuning_grid <- grid_regular(neighbors(), levels = 5)
knn_cv_results <- knn_workflow %>%
  tune_grid(resamples = folds,
            grid = knn_tuning_grid,
            metrics = metric_set(roc_auc))
stopCluster(cl)

# Get the best tuning parameters
knn_besttune <- knn_cv_results %>%
  select_best(metric = "roc_auc")

# fit and make predictions
knn_fit <- knn_workflow %>%
  finalize_workflow(knn_besttune) %>%
  fit(data = train_dirty)
knn_predictions <- predict(knn_fit,
                           new_data = test_dirty,
                           type = "prob")$.pred_1

# Write output
knn_output <- tibble(id = test_dirty$id,
                     Action = knn_predictions)
vroom_write(knn_output, "knn_model.csv", delim = ",")

#################
# Random Forest #
#################

# Create a model
forest_model <- rand_forest(min_n = tune()) %>%
  set_mode("classification") %>%
  set_engine("ranger")

# Create the workflow
forest_workflow() %>%
  add_model(forest_model) %>%
  add_recipe(recipe)

# Set up parallelization
registerDoParallel(cl)

# Tuning
forest_tuning_grid <- grid_regular(min_n(), levels = 20)
forest_cv_results <- forest_workflow %>%
  tune_grid(resamples = folds,
            grid = forest_tuning_grid,
            metrics = metric_set(roc_auc))

# Get the best tuning parameters
forest_besttune <- forest_cv_results %>%
  select_best(metric = "roc_auc")

# Fit and make predictions
forest_fit <- forest_workflow %>%
  finalize_workflow(forest_besttune) %>%
  fit(data = train_dirty)
forest_predictions <- predict(forest_fit,
                              new_data = test_dirty,
                              type = "prob")$.pred_1

# Write output
forest_output <- tibble(id = test_dirty$id,
                        Action = forest_output)
vroom_write(forest_output, "random_forest.csv", delim = ",")

