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

6. Test streamlit using app.py

7. Test software using run.py

8. Once all is good and there are no issues we can run
- ```pyinstaller --onedir --windowed run.py```
