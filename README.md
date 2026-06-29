# vehicle-analytics-ml

To run locally:

1. Build a venv if you do not have one:
- Create the venv on PowerShell
- ``` python -m venv venv ```

2. Activate the venv
- ``` venv/Scripts/Activate ```

3. Run the requirements.txt
- only do this if you adding new packages or are making the venv for the first time
- ``` pip install -r requirements.txt ```

4. Make sure you are using the correct interpreter
- On VS code select crtl + shirt + p and python: select interpreter
- Use the venv/Scripts/python.exe

5. Run python scripts using python <name>.py

6. Test PySide6 desktop application using app.py `python app.py`

7. Test software using run.py

8. Once all is good and there are no issues we can run
- ```pyinstaller --onedir --windowed run.py```

--- Dev Test branch work

1. Phase 1 - Implementation

> [!NOTE] 
> When a task is completed please use `~~ Word ~~` for a strike through so we know what has been completed.

~~- Error when loading an icp file that has sheets or other datasets~~
    ~~- Error reads as no module named py arrow when uploading~~
    ~~- Likely causes are the JSON or the upload function~~
~~- Need to implement crtl+s and a save button to save the process~~
    ~~- If a user saves with having created a project a window should prompt to enter the project name~~
    ~~- If a user closes the application without saving or creating a project, need a prompt window as well~~
    ~~- If a user saves a project with the same name as another need to prompt for an override~~
    ~~- Need to include exporting~~
- Look into removing duplicates and missing cells
    - When you highlight the metric on the right table the rows should be highlighted
    - Right click the metric to go next and have a remove option
        - A window will pop up that shows the rows with missing cells and duplicates (missing cells should be highlighted per cell)
        - Have multiselect for removing and prompt to make sure the user is sure
- Noticed loading a large dataset that had 1.1 million rows was slow, that is expected but maybe look into optimization (parallel processing) when loading
- Noticed that when you update columns in a dataset and then upload another dataset only 1 column appears
    - Sometimes when you reopen the same dataset it has the same columns selected from the previously opened dataset, same applies when you load an icp process
- Change the row limit to 1 thousand and allow to click through, 1-1000 -> 1001-2000
- Add a right click or drop down to each columns missing values for imputation (mean, median, mode, custom)
    - If custom is used on a previously int, float, or binary and the user chooses a string the columns dtype needs to be updates
    - Should be able to remove imputation
- Add a select all columns option for features and add columns from the original df to analysis
    - Columns in the feature engineering should be divided by categorical and numerical
- Change the colours for the heatmap and add tick markers in the heatmap
    - The heatmap takes one value less, it will not analyze the label column
- The label options should only show a drop down of the selected columns, it should be blank initially and provide a none option for semi-supervised and unsupervised learning
    - The label column when selected should be highlighted
- Remove scroll bar on file summary going horizontally
- Allow for window resizing on the missing value summary to make it longer
- Scrollbar should start the beginning of the column not at the header
- Visualization tab should update the columns based on required data types
- Analysis and visualization are using the previous project and need to be reset when a new dataset is loaded, this correlates to the pop up window for choosing a new dataset
- For feature extraction separate columns by numeric and non-numeric
- Select feature should have a drop down for numeric and non-numeric
- Have use raw dataset feature for comparison
- Standardize and normalize need to implemented, maybe by applying right click on the column in missing value summary or dataset preview
