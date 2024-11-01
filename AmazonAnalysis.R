# Libraries
library(tidyverse)
library(tidymodels)
library(discrim)
library(vroom)
library(doParallel)
library(themis)

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
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.9) %>%
  step_smote(all_outcomes(), neighbors = 10)
prepped_recipe <- prep(recipe)
#clean_data <- bake(prepped_recipe, new_data = train_dirty)

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
cl <- makePSOCKcluster(num_cores)
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
cl <- makePSOCKcluster(num_cores)
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
forest_workflow <- workflow() %>%
  add_model(forest_model) %>%
  add_recipe(recipe)

# Set up parallelization
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

# Tuning
forest_tuning_grid <- grid_regular(min_n(), levels = 10)
forest_cv_results <- forest_workflow %>%
  tune_grid(resamples = folds,
            grid = forest_tuning_grid,
            metrics = metric_set(roc_auc))
stopCluster(cl)

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
                        Action = forest_predictions)
vroom_write(forest_output, "random_forest.csv", delim = ",")

###############
# Naive Bayes #
###############

# Create a model
bayes_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

# Create the workflow
bayes_workflow <- workflow() %>%
  add_model(bayes_model) %>%
  add_recipe(recipe)

# Set up parallelization
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

# Tuning
bayes_tuning_grid <- grid_regular(Laplace(), smoothness(), levels = 20)
bayes_cv_results <- bayes_workflow %>%
  tune_grid(resamples = folds,
            grid = bayes_tuning_grid,
            metrics = metric_set(roc_auc))
stopCluster(cl)

# Get the best tuning parameters
bayes_besttune <- bayes_cv_results %>%
  select_best(metric = "roc_auc")

# Fit and make predictions
bayes_fit <- bayes_workflow %>%
  finalize_workflow(bayes_besttune) %>%
  fit(data = train_dirty)
bayes_predictions <- predict(bayes_fit,
                             new_data = test_dirty,
                             type = "prob")$.pred_1

# Write output
bayes_output <- tibble(id = test_dirty$id,
                       Action = bayes_predictions)
vroom_write(bayes_output, "naive_bayes.csv", delim = ",")

##############
# Linear SVM #
##############

# Create a model
linear_svm_model <- svm_linear(cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Create the workflow
linear_svm_workflow <- workflow() %>%
  add_model(linear_svm_model) %>%
  add_recipe(recipe)

# Set up parallelization
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

# Tuning
linear_svm_tuning_grid <- grid_regular(cost(), levels = 10)
linear_svm_cv_results <- linear_svm_workflow %>%
  tune_grid(resamples = folds,
            grid = linear_svm_tuning_grid,
            metrics = metric_set(roc_auc))
stopCluster(cl)

# Get the best tuning parameters
linear_svm_besttune <- linear_svm_cv_results %>%
  select_best(metric = "roc_auc")

# Fit and make predictions
linear_svm_fit <- linear_svm_workflow %>%
  finalize_workflow(linear_svm_besttune) %>%
  fit(data = train_dirty)
linear_svm_predictions <- predict(linear_svm_fit,
                                  new_data = test_dirty,
                                  type = "prob")$.pred_1

# Write output
linear_svm_output <- tibble(id = test_dirty$id,
                            Action = linear_svm_predictions)
vroom_write(linear_svm_output, "linear_svm.csv", delim = ",")

##############
# Radial SVM #
##############

# Create a model
radial_svm_model <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Create the workflow
radial_svm_workflow <- workflow() %>%
  add_model(radial_svm_model) %>%
  add_recipe(recipe)

# Set up parallelization
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

# Tuning
radial_svm_tuning_grid <- grid_regular(cost(), rbf_sigma(), levels = 10)
radial_svm_cv_results <- radial_svm_workflow %>%
  tune_grid(resamples = folds,
            grid = radial_svm_tuning_grid,
            metrics = metric_set(roc_auc))
stopCluster(cl)

# Get the best tuning parameters
radial_svm_besttune <- radial_svm_cv_results %>%
  select_best(metric = "roc_auc")

# Fit and make predictions
radial_svm_fit <- radial_svm_workflow %>%
  finalize_workflow(radial_svm_besttune) %>%
  fit(data = train_dirty)
radial_svm_predictions <- predict(radial_svm_fit,
                                  new_data = test_dirty,
                                  type = "prob")$.pred_1

# Write output
radial_svm_output <- tibble(id = test_dirty$id,
                            Action = radial_svm_predictions)
vroom_write(radial_svm_output, "radial_svm.csv", delim = ",")

##############
# poly SVM #
##############

# Create a model
poly_svm_model <- svm_poly(cost = tune(), degree = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Create the workflow
poly_svm_workflow <- workflow() %>%
  add_model(poly_svm_model) %>%
  add_recipe(recipe)

# Set up parallelization
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

# Tuning
poly_svm_tuning_grid <- grid_regular(cost(), degree(), levels = 10)
poly_svm_cv_results <- poly_svm_workflow %>%
  tune_grid(resamples = folds,
            grid = poly_svm_tuning_grid,
            metrics = metric_set(roc_auc))
stopCluster(cl)

# Get the best tuning parameters
poly_svm_besttune <- poly_svm_cv_results %>%
  select_best(metric = "roc_auc")

# Fit and make predictions
poly_svm_fit <- poly_svm_workflow %>%
  finalize_workflow(poly_svm_besttune) %>%
  fit(data = train_dirty)
poly_svm_predictions <- predict(poly_svm_fit,
                                  new_data = test_dirty,
                                  type = "prob")$.pred_1

# Write output
poly_svm_output <- tibble(id = test_dirty$id,
                            Action = poly_svm_predictions)
vroom_write(poly_svm_output, "poly_svm.csv", delim = ",")

