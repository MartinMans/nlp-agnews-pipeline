from src.inference.baseline import predict

if __name__ == "__main__":
    sample_text = "Stock markets rally as tech companies report strong earnings."
    result = predict(sample_text)

    print("Input:", sample_text)
    print("Prediction:", result)
