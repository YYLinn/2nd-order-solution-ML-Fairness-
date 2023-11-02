    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score, recall_score, f1_score
    import tensorflow as tf
    from sklearn.metrics import log_loss
    from sklearn.metrics import roc_auc_score
    import seaborn as sns
    import numpy as np
    from scipy.stats import gaussian_kde




    def plot_distributions(y, Z, fname=None):
        fig, ax = plt.subplots(figsize=(5, 4))
        
        male_data = y[Z['SEX_Male'] == 1].values.reshape(-1)  # Ensuring it's a 1D array
        female_data = y[Z['SEX_Female'] == 1].values.reshape(-1)  # Ensuring it's a 1D array
        
        # Compute KDE for male and female data
        kde_male = gaussian_kde(male_data)
        kde_female = gaussian_kde(female_data)
        
        x = np.linspace(0, 1, 1000)  # Define range for x-axis
        ax.plot(x, kde_male(x), label='Male', color='blue')
        ax.plot(x, kde_female(x), label='Female', color='green')  # Change to 'pink' if you want
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 7)
        ax.set_yticks([])
        ax.set_title("Sensitive attribute: SEX is not included")
        ax.set_ylabel('prediction distribution')
        ax.set_xlabel(r'$P({{outcome=1}}|SEX)$')
        ax.legend(loc='upper right')

        fig.tight_layout()
        if fname is not None:
            plt.savefig(fname, bbox_inches='tight', dpi=300)
            
        plt.show()
        return fig



    def p_rule(y_pred, z_values, threshold=0.5):
        y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
        y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
        odds = y_z_1.mean() / y_z_0.mean()
        return np.min([odds, 1/odds]) * 100



    def train_model(X, y, categorical_cols, protected_df, protected_col):
        # create one hot encoding of categorical variables
        X = pd.get_dummies(X, columns=categorical_cols)
        print(X.columns)
        # enocde target variable
        y = np.where(y == "Yes", 1, 0)
        # encode protected attribute
        protected_df= pd.get_dummies(protected_df, columns=protected_col)
        # Get protected attributes (sex_male and sex_female)

        Z = protected_df
        print("OKay")
        # print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
        # print(f"targets y: {y.shape[0]} samples")
        # print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(
            X, y, Z, test_size=0.5, random_state=24
        )
        # Initialize the XGBoost classifier
        model = xgb.XGBClassifier()
        # Train the model on the training data
        model.fit(X_train, y_train)
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        # Print out the accuracy score
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        # other metrics
        print("Precision: ", precision_score(y_test, y_pred))
        print("Recall: ", recall_score(y_test, y_pred))
        print("F1: ", f1_score(y_test, y_pred))
        # calculate roc auc score
        # fig = plot_distributions(y_pred, Z_test, fname='images/biased_training.png')
        # plt.show()
        # plot distributionns of prediction with respect to the protected attribute
        # In the train_model function, after the line where y_pred is computed, add:
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # This line was already there
        print("ROC AUC: ", roc_auc_score(y_test, y_pred_prob))
        # Convert y_pred to a DataFrame (so we can use it as input for plot_distributions)
        y_pred_df = pd.DataFrame(y_pred, columns=["prediction"])
        y_pred_df = y_pred_df.reset_index(drop=True)
        Z_test = Z_test.reset_index(drop=True)
        # After the model training in your main block, call the plot_distributions function:
        fig = plot_distributions(y_pred_df, Z_test, fname = None)
        # show the figure
        plt.show()
        print("The classifier satisfies the following %p-rules:")
        print(f"\tgiven attribute marriage;  {p_rule(y_pred, Z_test['MARRIAGE_Other']):.0f}%-rule")

        # # calculate feature importance
        # idx = np.argsort(model.feature_importances_)
        # plt.barh(X.columns[idx], model.feature_importances_[idx])
        # plt.xlabel("Xgboost Feature Importance")
        # plt.show()
        # importance = model.feature_importances_
        # # print name of most important features
        # print(X.columns[np.argsort(importance)[-5:][::-1]])
        return model


    # import tensorflow as tf
    # from sklearn.metrics import log_loss

    # # Define the adversary model
    # def build_adversary():
    #     model = tf.keras.Sequential()
    #     model.add(tf.keras.layers.Dense(32, activation='relu'))
    #     model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    #     model.compile(optimizer='adam', loss='binary_crossentropy')
    #     return model


    # for epoch in range(5):
        

    #     # Train primary model (XGBoost in this case)
    #     model_protected = train_model(X_protected, y, categorical_cols)

    #     # Generate predictions from primary model
    #     y_pred_prob = model_protected.predict_proba(X_protected)[:,1]
        
    #     # Train adversary to predict the protected attribute based on the primary model's predictions
    #     adversary_loss = adversary.train_on_batch(y_pred_prob, X_protected['SEX'])

    #     adversary_pred = adversary.predict(y_pred_prob)
    #     regularization_loss = log_loss(X_protected['SEX'], adversary_pred)

    #     print(f"Epoch {epoch + 1}/{num_epochs}, Adversary Loss: {adversary_loss:.4f}, Regularization Loss: {regularization_loss:.4f}")





    if __name__ == "__main__":
        data = pd.read_csv("../data/default_clean_v1.csv")
        data = data.drop(columns=["Unnamed: 0", "default payment next month"], axis=1)
        # # check if 9 is present in PAY_0 or PAY_2
        # print(data[(data["PAY_0"] == 9) | (data["PAY_2"] == 9)])
        X_protected = data.drop(["default", "ID"], axis=1)
        y = data["default"]
        protected_attr = ["SEX", "MARRIAGE"]
        prot_df = X_protected[protected_attr]
        X_protected = X_protected.drop( protected_attr, axis=1)
        cat_col_prot = ["SEX", "MARRIAGE"]
        categorical_cols = [ "EDUCATION"]
        model_protected = train_model(X_protected, y, categorical_cols, prot_df, cat_col_prot)

    # X = data.drop(["default", "ID", "SEX"], axis=1)
    # protected_attr = "SEX" 
    # categorical_cols = ["EDUCATION", "MARRIAGE"]
    # model = train_model(X, y, categorical_cols)
    # # plot distributionns of prediction with respect to the protected attribute
    # y_pred = model.predict_proba(X)[:,1]


    # adversary = build_adversary()
    # num_epochs = 5
    # for epoch in range(num_epochs):
    #     # Train primary model (XGBoost in this case)
    #     model = train_model(X, y, categorical_cols)

    #     # Generate predictions from primary model
    #     y_pred_prob = model.predict_proba(X)[:,1]
    
    #     # Train adversary to predict the protected attribute based on the primary model's predictions
    #     adversary_loss = adversary.train_on_batch(y_pred_prob, X_protected['SEX'])

    #     adversary_pred = adversary.predict(y_pred_prob)
    #     regularization_loss = log_loss(X_protected['SEX'], adversary_pred)

    #     print(f"Epoch {epoch + 1}/{num_epochs}, Adversary Loss: {adversary_loss:.4f}, Regularization Loss: {regularization_loss:.4f}")








