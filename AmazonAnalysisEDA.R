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

# Attempt to look at correlations between some of the biggest groups
no_resp <- clean_data[-10]
preds <- no_resp %>%
  colnames()
response <- tibble(Column = character(),
                   Value = integer(),
                   Count = integer(),
                   Prop = double())

for (pred in preds) {
  counts <- no_resp[pred] %>%
    table() %>%
    as.double()
  cols <- rep(pred, times = length(counts))
  values <- no_resp %>%
    .[[pred]] %>%
    levels()
  props <- c()

  i <- 0
  for (val in values) {
    i <- i + 1
    prop <- clean_data[clean_data[pred] == val, ] %>%
      .[["ACTION"]] %>%
      mean()
    props[i] <- prop
  }
  tmp_df <- tibble(Column = cols,
                   Value = values,
                   Count = counts,
                   Prop = props)
  response <- rbind(response, tmp_df)
}

to_compare <- response[response["Value"] > 150, ]
compare <- function(column, value, comparison) {
}

