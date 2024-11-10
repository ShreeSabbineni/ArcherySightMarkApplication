from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Collect initial input data from the user
print("Enter distances and sight markings for archery (e.g., distance 20 with marking 15).")
print("Type 'done' when you have entered all data.\n")

distances = []
sight_markings = []

# Prompt the user for input
while True:
    distance = input("Enter distance (or type 'done' to finish): ")
    if distance.lower() == 'done':
        break
    sight_mark = input("Enter sight marking for distance " + distance + ": ")
    distances.append(float(distance))
    sight_markings.append(float(sight_mark))

# Check if there is enough data
if len(distances) < 2:
    print("Please enter at least two distance-marking pairs.")
else:
    # Prepare data for model training
    X = np.array(distances).reshape(-1, 1)  # Reshape to 2D array for sklearn
    y = np.array(sight_markings)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the slope and intercept of the fitted line
    print(f"\nLine of best fit: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

    # Prompt user for a new distance and predict the sight mark
    new_distance = float(input("\nEnter a new distance to get the predicted sight mark: "))
    predicted_marking = model.predict(np.array([[new_distance]]))[0]
    
    print(f"Predicted sight mark for distance {new_distance} is: {predicted_marking:.2f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot actual sight markings as scatter points
    plt.scatter(distances, sight_markings, color='blue', label='Actual Sight Marks')

    # Plot the regression line
    x_range = np.linspace(min(distances), max(distances), 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    plt.plot(x_range, y_range, color='green', label='Regression Line (Best Fit)')

    # Plot the predicted sight mark in a different color
    plt.scatter(new_distance, predicted_marking, color='red', marker='x', s=100, label='Predicted Sight Mark')

    # Adding labels and legend
    plt.xlabel("Distance")
    plt.ylabel("Sight Mark")
    plt.title("Archery Sight Marks and Regression Line")
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()
