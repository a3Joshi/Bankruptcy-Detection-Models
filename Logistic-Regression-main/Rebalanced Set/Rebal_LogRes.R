# Load required libraries
library(glmnet)
library(readxl)
library(caret)
library(pROC)
library(MLmetrics)
# Load the training data
train_data <- read_excel("train_bal.xlsx")

# Convert 'Bankrupt' to a factor with valid level names
train_data$Bankrupt <- factor(train_data$Bankrupt, 
                              labels = make.names(levels(factor(train_data$Bankrupt))))

# Load the test data
test_data <- read_excel("test.xlsx")

# Convert 'Bankrupt' to a factor with valid level names
test_data$Bankrupt <- factor(test_data$Bankrupt, 
                              labels = make.names(levels(factor(test_data$Bankrupt))))


# Define the control method for training
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, 
                     summaryFunction = twoClassSummary)

# Define a grid of hyperparameters to tune
# Here, we're tuning the cost parameter of the logistic regression model
grid <- expand.grid(.alpha = seq(0, 1, length.out = 10),  # alpha corresponds to the type of regularization (0 for L2, 1 for L1)
                    .lambda = seq(0.001, 1, length.out = 100))  # lambda corresponds to the strength of regularization

# Train the model with hyperparameter tuning
set.seed(123)
model <- train( Bankrupt ~ . , data = train_data, method = "glmnet" ,
               trControl = ctrl, metric = "ROC" , tuneGrid = grid )

# Print the final model
print(model)

# Predict probabilities on the validation set
predictions <- predict(model, newdata = test_data)
prob <- predict(model, newdata = test_data , type = "prob")[,2]
# Calculate the ROC curve
roc_obj <- roc(test_data$Bankrupt, prob)

# Calculate the coordinates of the ROC curve
roc_coords <- coords(roc_obj, "best")

# The optimal threshold is the 'best' threshold
optimal_threshold <- roc_coords["threshold"]
optimal_threshold <- as.numeric(optimal_threshold)
# Make predictions based on the decision threshold
predictions <- ifelse(prob > optimal_threshold, 1, 0)

# Convert 'Bankrupt' to a factor with valid level names
predictions <- factor(predictions, 
                                labels = make.names(levels(factor(predictions))))
# Print the confusion matrix
confusionMatrix(predictions, test_data$Bankrupt)

f1 <- F1_Score(test_data$`Bankrupt?`, predictions)
f1

