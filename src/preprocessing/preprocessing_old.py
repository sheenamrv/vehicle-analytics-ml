from sklearn.preprocessing import StandardScaler, MinMaxScaler

def show_missing_values(df):

    print("Missing Values")

    missing = df.isnull().sum()

    for col, count in missing.items():

        if count > 0 : 
            print(f"{col} : {count}")

def fill_missing_values(df):

    method = None
    
    print("Columns")

    for col in df.columns:
        print(col)

    column = input("Column to fill ").strip()

    print("\nMethods:")
    print("1 - Mean")
    print("2 - Median")
    print("3 - Mode")
    print("4 - Constant")

    choice = input("Choice: ")

    if choice == "1":

        method = "mean"
        df[column] = df[column].fillna(
            df[column].mean()
        )

    elif choice == "2":

        method = "median"

        df[column] = df[column].fillna(
            df[column].median()
        )

    elif choice == "3":

        method = "mode"
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

    elif choice == "4":

        value = input(
            "Constant value: "
        )

        method = "constant"
        df[column] = df[column].fillna(
            value
        )

    return df, column, method

def standardize_col(df):

    print("Numeric Col: ")

    numeric_cols = list(df.select_dtypes(include="number").columns)

    for col in numeric_cols:
        print(col)

    cols = input("Col to change: ")

    selected = [c.strip() for c in cols.split(",")]

    mean = input("Enter the mean to change T/F: ")

    if mean is None:
        mean = None
    if mean in ['true', 't', '1', 'yes', 'y']:
        mean = True
    elif mean in ['false', 'f', '0', 'no', 'n']:
        mean=False

    std = input("Enter the std T/F: ")

    if std is None:
        std = None
    if std in ['true', 't', '1', 'yes', 'y']:
        std = True
    elif std in ['false', 'f', '0', 'no', 'n']:
        std=False

    scaler = StandardScaler(with_mean=bool(mean), with_std=bool(std))

    df[selected] = scaler.fit_transform(df[selected])

    return df, selected, mean, std

def normalize_col(df):

    print("Numeric Col: ")

    numeric_cols = list(df.select_dtypes(include="number").columns)

    for col in numeric_cols:
        print(col)

    cols = input("Col to change: ")

    selected = [c.strip() for c in cols.split(",")]

    scaler = MinMaxScaler()

    df[selected] = scaler.fit_transform(df[selected])

    return df, selected