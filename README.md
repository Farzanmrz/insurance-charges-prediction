# Insurance Charges Prediction

This project demonstrates the application of closed form linear regression on the insurance dataset. The goal is to predict insurance charges based on various features such as age, sex, BMI, number of children, smoking status, and region. The notebook includes preprocessing steps, training and testing the model, and performing cross-validation.

## Project Structure

- `main.ipynb`: Jupyter Notebook containing the step-by-step implementation.
- `main.py`: Python script generated from the notebook.
- `insurance.csv`: Dataset containing insurance data.

## Running the Project

### Setup

To set up the environment, ensure you have the required libraries. You can install them using pip:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Using Jupyter Notebook

1. Open the `main.ipynb` file in Jupyter Notebook.
2. Run all cells to execute the code step-by-step.

### Using the Python Script

1. Ensure you have the `insurance.csv` file in the same folder as the script.
2. Run the script using the following command:

```bash
python main.py
```

## Expected Output

The project will provide the RMSE and SMAPE values for both training and validation sets after training the linear regression model. It will also perform cross-validation with different values of S (number of folds) and print the mean and standard deviation of the RMSE values.

### Example Output

```bash
Training RMSE: 6164.43, Validation RMSE: 5800.88
Training SMAPE: 19.22%, Validation SMAPE: 17.61%

Cross-Validation Results:
Mean RMSE for S = 3: 6105.73
Standard Deviation for S = 3: 24.22

Mean RMSE for S = 223: 6087.05
Standard Deviation for S = 223: 1.23

Mean RMSE for S = 1338: 6087.39
Standard Deviation for S = 1338: 0.00
```

## Future Work

- Explore more sophisticated regression models such as Ridge Regression and Lasso Regression.
- Perform feature engineering to improve the model's predictive performance.
- Include more detailed error analysis and visualization of results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact 

Farzan Mirza: [farzanmrz@gmail.com](mailto:farzanmrz@gmail.com) | [farzan.mirza@drexel.edu](mailto:farzan.mirza@drexel.edu) | [LinkedIn](https://www.linkedin.com/in/farzan-mirza13/)

