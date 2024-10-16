# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(doParallel)

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
num_cores <- 4
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

