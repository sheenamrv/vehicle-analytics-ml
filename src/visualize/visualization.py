import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def export_plot_image(fig, path):
    # Ensure the latest interactive canvas state is flushed before export.
    if getattr(fig, "canvas", None) is not None:
        fig.canvas.draw()
    fig.savefig(path, dpi=200, bbox_inches="tight")


def create_visualization(df):

    print("\nVisualization Types")
    print("1 - Scatter Plot")
    print("2 - Line Plot")
    print("3 - Histogram")
    print("4 - Box Plot")
    print("5 - Bar Chart")

    choice = input("\nChoice: ").strip()

    if choice == "1":
        x, y = show_col(df, choice)

        scatter_plot(df, x, y)

    # elif choice == "2":
    #     line_plot(df)

    elif choice == "3":
        col = show_col(df, choice)
        histogram(df, col)

    # elif choice == "4":
    #     box_plot(df)

    # elif choice == "5":
    #     bar_chart(df)

    else:
        print("Invalid choice")

def show_col(df, choice):

    print("Columns: ")
    for col in df.columns:
        print(col)

    if choice == "3" or choice == "4":

        column = input("Column: ")
        return column
    
    else:

        x_col = input("X Col: ")
        y_col = input("y col: ")
        return x_col, y_col

def scatter_plot(df, x, y):

    plt.figure(figsize=(8,5))

    plt.scatter(df[x], df[y])

    plt.xlabel(x)
    plt.ylabel(y)

    plt.title(f"{y} vs. {x}")

    plt.show()

def histogram(df, col):

    plt.figure(figsize=(8,5))

    df[col].hist()

    plt.xlabel(col)
    plt.axvline(np.mean(df[col]), color='red', linestyle='dashed')

    plt.title(f"Histogram: {col}")

    plt.show()