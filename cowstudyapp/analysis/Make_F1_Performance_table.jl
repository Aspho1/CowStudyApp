# State	Actual	Predicted	Accuracy	Precision	Specificity	Sensitivity 	F1
# Grazing	205	203	95.0%	94.6%	96.0%	93.7%	94.1%
# Resting	230	232	94.6%	94.0%	94.4%	94.8%	94.4%
# Traveling	46	46	98.8%	93.5%	99.3%	93.5%	93.5%

# Can we make a table like the above for an input csv file? 
# The input file will have atleast two columns (Actual and Predicted)
# classes should be recast from 0,1,2 to Grazing Resting Traveling. 

using CSV
using DataFrames
using StatsBase
using Printf
using LinearAlgebra

# Function to create confusion matrix
function confusion_matrix(actual, predicted)
    classes = unique([actual..., predicted...])
    n = length(classes)
    cm = zeros(Int, n, n)
    
    for i in 1:length(actual)
        actual_idx = findfirst(x -> x == actual[i], classes)
        predicted_idx = findfirst(x -> x == predicted[i], classes)
        cm[actual_idx, predicted_idx] += 1
    end
    
    return cm
end


# Function to display confusion matrix
function display_confusion_matrix(actual, predicted)
    # Get unique classes
    classes = unique([actual..., predicted...])
    sort!(classes)  # Sort classes for consistent display
    
    # Create the confusion matrix
    cm = zeros(Int, length(classes), length(classes))
    for i in 1:length(actual)
        actual_idx = findfirst(x -> x == actual[i], classes)
        predicted_idx = findfirst(x -> x == predicted[i], classes)
        cm[predicted_idx, actual_idx] += 1  # Note: rows=PREDICTION, columns=TRUTH
    end
    
    # Print the confusion matrix header
    println("\nConfusion Matrix (rows=PREDICTION, columns=TRUTH):")
    
    # Print column headers (TRUTH classes)
    print("            ")  # Space for row labels
    for class in classes
        print(lpad(string(class), 10))
    end
    println()
    
    # Print each row
    for (i, class) in enumerate(classes)
        print(rpad(string(class), 12))  # Row label (PREDICTION)
        for j in 1:length(classes)
            print(lpad(string(cm[i, j]), 10))
        end
        println()
    end
    
    return cm
end

# Function to calculate metrics for each class
function calculate_class_metrics(actual, predicted)
    # Create label encodings for confusion matrix
    classes = unique([actual..., predicted...])
    class_indices = Dict(class => i for (i, class) in enumerate(classes))
    
    # Encode actual and predicted
    actual_encoded = [class_indices[a] for a in actual]
    predicted_encoded = [class_indices[p] for p in predicted]
    
    # Create the confusion matrix
    cm = zeros(Int, length(classes), length(classes))
    for i in 1:length(actual_encoded)
        cm[actual_encoded[i], predicted_encoded[i]] += 1
    end
    
    # Calculate metrics for each class
    results = DataFrame(
        State = classes,
        Actual = zeros(Int, length(classes)),
        Predicted = zeros(Int, length(classes)),
        Accuracy = zeros(Float64, length(classes)),
        Precision = zeros(Float64, length(classes)),
        Specificity = zeros(Float64, length(classes)),
        Sensitivity = zeros(Float64, length(classes)),
        F1 = zeros(Float64, length(classes))
    )
    
    for (i, class) in enumerate(classes)
        # For binary case: this class vs all others
        tp = cm[i, i]                          # True positives
        fp = sum(cm[:, i]) - tp                # False positives
        fn = sum(cm[i, :]) - tp                # False negatives
        tn = sum(cm) - tp - fp - fn            # True negatives
        
        # Calculate metrics
        accuracy = (tp + tn) / sum(cm)
        precision = tp / (tp + fp)
        sensitivity = tp / (tp + fn)  # Recall
        specificity = tn / (tn + fp)
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
        
        # Store in results
        results[i, :Actual] = sum(cm[i, :])
        results[i, :Predicted] = sum(cm[:, i])
        results[i, :Accuracy] = accuracy * 100
        results[i, :Precision] = precision * 100
        results[i, :Specificity] = specificity * 100
        results[i, :Sensitivity] = sensitivity * 100
        results[i, :F1] = f1 * 100
    end
    
    overall_accuracy = sum(diag(cm)) / sum(cm) * 100
    
    # Calculate micro-averaged F1 (same as accuracy for multi-class)
    total_tp = sum(diag(cm))
    total_fp = sum(sum(cm, dims=1)) - total_tp
    total_fn = sum(sum(cm, dims=2)) - total_tp
    
    micro_precision = total_tp / (total_tp + total_fp)
    micro_recall = total_tp / (total_tp + total_fn)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) * 100
    
    return results, overall_accuracy, micro_f1

end

# Function to format and display the final table
function display_metrics_table(results, overall_accuracy, micro_f1)
    # Format the results for display
    formatted = copy(results)
    
    for col in [:Accuracy, :Precision, :Specificity, :Sensitivity, :F1]
        formatted[!, col] = [@sprintf("%.2f%%", value) for value in formatted[!, col]]
    end
    
    # Print the table header
    println("State\tActual\tPredicted\tAccuracy\tPrecision\tSpecificity\tSensitivity\tF1")
    
    # Print each row
    for row in eachrow(formatted)
        println("$(row.State)\t$(row.Actual)\t$(row.Predicted)\t$(row.Accuracy)\t$(row.Precision)\t$(row.Specificity)\t$(row.Sensitivity)\t$(row.F1)")
    end
  
    # Print overall metrics
    println("\nOverall Accuracy: $(@sprintf("%.1f%%", overall_accuracy))")
    println("Micro-Averaged F1: $(@sprintf("%.1f%%", micro_f1))")
        

    return formatted
end

# Main function to process the data and create the metrics table
function create_performance_table(file_path; class_map=nothing)
    # Read CSV file
    data = CSV.File(file_path) |> DataFrame

    # println(data)
    
    # Ensure required columns exist
    if !all(col -> col in names(data), ["Actual", "Predicted"])
        error("CSV must contain 'Actual' and 'Predicted' columns")
    end
     
    # If class map is provided, recast class numbers to names
    if class_map !== nothing
        data[!, :Actual] = map(x -> class_map[x], data[!, :Actual])
        data[!, :Predicted] = map(x -> class_map[x], data[!, :Predicted])
    end
    
    # Calculate metrics
    results, overall_accuracy, micro_f1 = calculate_class_metrics(data[!, :Actual], data[!, :Predicted])
    
    # Display the formatted table
    formatted_results = display_metrics_table(results, overall_accuracy, micro_f1)
    
    # Display confusion matrix
    confusion_mat = display_confusion_matrix(data[!, :Actual], data[!, :Predicted])
    
    return results, formatted_results, overall_accuracy, micro_f1, confusion_mat
end

# Example usage
# Define the class mapping
class_map = Dict(0 => "Grazing", 1 => "Resting", 2 => "Traveling")

# Replace with your actual file path
file_path = "O:/Education/CowStudyApp/data/cv_results/RB_22/LSTM/ops/lstm_cv_preds.csv"

# Create and display the performance table
results, formatted_results, overall_accuracy, micro_f1, confusion_mat = create_performance_table(file_path, class_map=class_map)

# The results are also available as a DataFrame for further processing

println(file_path)
display(results)
display(confusion_mat)
println("\nOverall Accuracy: $(@sprintf("%.2f%%", overall_accuracy))")
println("Micro-Averaged F1: $(@sprintf("%.2f%%", micro_f1))")