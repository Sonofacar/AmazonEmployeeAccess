# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)

# Read the data
train_dirty <- vroom("train.csv") %>%
  mutate(ACTION = factor(ACTION))
test_dirty <- vroom("test.csv")

# Make a recipe
recipe <- recipe(ACTION ~ ., data = train_dirty) %>%
  step_mutate_at(all_double_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors())
prepped_recipe <- prep(recipe)
clean_data <- bake(prepped_recipe, new_data = train_dirty)

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

