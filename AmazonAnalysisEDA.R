# Load libraries
library(tidyverse)
library(tidymodels)
library(vroom)

# Read the data
data <- vroom("train.csv")

# Make a recipe
recipe <- recipe(ACTION ~ ., data = data) %>%
  step_mutate_at(all_double_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors())
prepped_recipe <- prep(recipe)
clean_data <- bake(prepped_recipe, new_data = data)

# Data Dimensions
clean_data %>%
  dim()

# Making a more reasonable dataframe
recipe <- recipe(ACTION ~ ., data = data) %>%
  step_mutate_at(all_double_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001)
prepped_recipe <- prep(recipe)
clean_data <- bake(prepped_recipe, new_data = data)

# Check for relationships between predictors
response <- tibble(Predictor = factor(),
                   Value = character(),
                   Relation_with = character(),
                   Relation_value = factor(),
                   Difference = double(),
                   Magnitude = integer())
no_resp <- clean_data[-10]
predictors <- no_resp %>%
  colnames()
factors <- list()

# Cycle through predictors
for (pred in predictors) {
  print(pred)
  factors[[pred]] <- no_resp[[pred]] %>%
    levels()
}

# Now, using this list, cycle through it all to
# find correlations between factors of predictors
rows <- 0
pred <- 0
n_pred <- length(predictors)

for (f in factors) {
  pred <- pred + 1
  current_pred <- predictors[pred]

  for (fact in f) {
    value <- fact
    rels <- predictors[pred != 1:n_pred]
    n_rel <- 1

    for (other in factors[-pred]) {
      rel <- rels[n_rel]

      for (rel_val in other) {
        total <- dim(no_resp)[1]
        sub_total <- no_resp[no_resp[current_pred] == fact, ] %>%
          dim() %>%
          .[1]
        rel_total <- no_resp[no_resp[rel] == rel_val, ] %>%
          dim() %>%
          .[1]
        rel_sub_total <- no_resp[(no_resp[current_pred] == fact) &
                                   (no_resp[rel] == rel_val), ] %>%
          dim() %>%
          .[1]
        diff <- (rel_total / total) - (rel_sub_total / sub_total)
        mag <- sub_total
        row <- list(Predictor = current_pred,
                    Value = value,
                    Relation_with = rel,
                    Relation_value = rel_val,
                    Difference = diff,
                    Magnitude = mag)
        rows <- rows + 1
        response[rows, ] <- row
      }
    }
  }
}

