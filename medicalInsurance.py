import pandas as pd 
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm  import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def importData(currentPath:str) -> pd.DataFrame | None:
    dataFrame = None
    try:
        currentPath = currentPath + "/" + "medical Insurance cost prediction.csv"
        dataFrame = pd.read_csv(currentPath)
    except Exception as e:
        print(f"Exception inside when importData : {e}")
    return dataFrame


def exploreEDA(medicalRecords: pd.DataFrame):
    try:
        print('######################## INFO ########################')
        print(medicalRecords.info())

        print('######################## HEAD ########################')
        print(medicalRecords.head())

        print('######################## TAIL ########################')
        print(medicalRecords.tail())
        
        print('######################## SAMPLE ########################')
        print(medicalRecords.sample(5))


        print('######################## FIND THE NAN VALUE ########################')
        print(medicalRecords.isnull().sum())


        print("######################## MANAGE THE DUPLICATE VALUES ########################")
        print(f"Duplicate values = ",medicalRecords.duplicated().sum())
        medicalRecords.drop_duplicates(inplace=True)
        print(f"Duplicate values = ",medicalRecords.duplicated().sum())

        
        print("######################## DESCRIPTIVE STATISTICS ANALYSIS ########################")

        pd.options.display.float_format = '{:.2f}'.format
        print(medicalRecords.describe(include=['int64','float64']).T)

        
        print("######################## CORR ########################")
        tempMedical = medicalRecords
        corrMedical = tempMedical.drop(['sex','smoker', 'region'],axis=1)
        print(corrMedical.corr())




        #     print("######################## PLOT NUMERICAL ########################")
        #     numerical_cols = ["age", "bmi", "children", "charges"]

        #     fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        #     axes = axes.flatten()

        #     for ax, col in zip(axes, numerical_cols):
        #         sns.histplot(
        #             data=medicalRecords, x=col, kde=(col != "children"),
        #             bins=20, ax=ax
        #         )
        #         ax.set_title(f"Distribution of {col}")

        #     plt.tight_layout()
        #     plt.show()


            
        #     print("######################## PLOT CATEGORICAL ########################")
        #     categorical_cols = ["sex", "smoker", "region"]

        #     fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        #     for ax, col in zip(axes, categorical_cols):
        #         sns.countplot(data=medicalRecords, x=col, hue=col,
        #                     legend=False, ax=ax)
        #         ax.set_title(f"Count of {col}")
        #         ax.tick_params(axis="x", rotation=25)

        #     plt.tight_layout()
        #     plt.show()



        #     print("######################## CORRELATION HEATMAP ########################")
        #     corr = medicalRecords[numerical_cols].corr(method="pearson")

        #     plt.figure(figsize=(8, 6))
        #     sns.heatmap(
        #         corr,
        #         annot=True,
        #         cmap="coolwarm",
        #         fmt=".2f",
        #         vmin=-1,
        #         vmax=1
        #     )
        #     plt.title("Correlation Heatmap")
        #     plt.tight_layout()
        #     plt.show()

        #     print("Correlation with charges:")
        #     print(corr["charges"].sort_values(ascending=False))


    except Exception as e:
        print(f"Exception inside exploreEDA Part {e}")



def outlierTreatment(records : pd.DataFrame) -> pd.DataFrame | None:
    medicalRecords = None
    try:
        numerical_cols = ["age", "bmi", "children", "charges"]

        df_clean = records.copy()
        outlier_mask = pd.Series(False, index=df_clean.index)

        for col in numerical_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outlier_mask |= (
                (df_clean[col] < lower) |
                (df_clean[col] > upper)
            )

        print("Rows before removal:", len(df_clean))
        print("Outlier rows to remove:", outlier_mask.sum())

        medicalRecords = df_clean.loc[~outlier_mask].copy()



    except Exception as e:
        print(f"Exception inside the outlierTreatment {e}")


    return medicalRecords


def featureEngineering(records: pd.DataFrame) -> pd.DataFrame | None:
    medicalRecords = None
    try:
        # Binary encoding
        records["sex"] = records["sex"].map({
            "female": 0,
            "male": 1
        })

        records["smoker"] = records["smoker"].map({
            "no": 0,
            "yes": 1
        })


        # One-hot encode region
        medicalRecords = pd.get_dummies(
            records,
            columns=["region"],
            dtype=int,
            drop_first=True
        )

        print(records.head())


        
        skew_cols = ["age", "bmi", "children", "charges"]

        print("Skewness before treatment:")
        print(medicalRecords[skew_cols].skew().sort_values(ascending=False))

        # Check skewness before transformation
        print("Charges skewness before:", medicalRecords["charges"].skew())

        # Transform the target if appropriate
        medicalRecords["charges_log"] = np.log1p(medicalRecords["charges"])

        print("Charges skewness after:", medicalRecords["charges_log"].skew())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(medicalRecords["charges"], kde=True, ax=axes[0])
        axes[0].set_title("Original Charges")

        sns.histplot(medicalRecords["charges_log"], kde=True, ax=axes[1])
        axes[1].set_title("Log-Transformed Charges")

        plt.tight_layout()
        plt.show()










    except Exception as e:
        print(f"Exception Inside featureEngineering {e}")

    return medicalRecords


def models(records: pd.DataFrame):
    try:
        x = records.drop(['charges','charges_log'],axis=1)
        y = records['charges']
       
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Scale the features
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        model_list= [
            LinearRegression(),
            Ridge(),
            Lasso(),
            ElasticNet(),
            DecisionTreeRegressor(),
            RandomForestRegressor(),
            GradientBoostingRegressor(),
            KNeighborsRegressor(),
            XGBRegressor(),
            SVR()
        ]
        count = 0
        
        results = []
        for model in model_list:
            model_test = model
            model_test.fit(X_train,y_train)
            y_pred = model_test.predict(X_test)
            count += 1
            # Evaluate

            results.append({
                'Model': model.__class__.__name__,
                "Score": model_test.score(X_train, y_train),
                "R2:": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
            })
            

        results_df = pd.DataFrame(results)
        print(results_df)
    except Exception as e:
        print(f"Exception inside the models {e}")

def model(records: pd.DataFrame):
    try:
        x = records.drop(['charges','charges_log'],axis=1)
        y = records['charges']
       
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Scale the features
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        model = RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1)


        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        # Evaluate

        print(f"Score: {model.score(X_train, y_train)}")
        print(f"R2 {r2_score(y_test, y_pred)}")
        print(f"MAE {mean_absolute_error(y_test, y_pred)}")
        print(f"RMSE : { np.sqrt(mean_squared_error(y_test, y_pred))}")

         # Save both fitted objects
         
        currentPath = os.path.dirname(__file__)
        joblib.dump(scaler, f"{currentPath}//scaler.pkl")
        joblib.dump(model, f"{currentPath}//model.pkl")

        print("Scaler and model saved successfully.")
    

    except Exception as e:
        print(f"Exception inside the model {e}")




if __name__ == "__main__":
    currentPath = os.path.dirname(__file__)
    records = importData(currentPath)
    if records is None:
        print("Medical Records Is None")
    else:
        exploreEDA(medicalRecords = records )
        records = outlierTreatment(records= records)

        if records is None:
            print("Records is None After outlier")
        else:
            records = featureEngineering(records= records)

            models(records=records)

            model(records)




