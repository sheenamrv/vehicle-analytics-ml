# Interactive Machine Learning Application

A desktop machine learning application for vehicle analytics built with **Python** and **PySide6**.

## Project Resources

### Figma Design

View the project design here:

**[Figma Design](https://www.figma.com/design/PT2GGS6hPYO4dnTnLpy49s/BISI-Capstone-UI-Design-2026?node-id=0-1&t=77XuAQU0iZNhR8yk-1)**

### Figma Prototype

View the interactive prototype here:

**[Figma Prototype](https://www.figma.com/proto/PT2GGS6hPYO4dnTnLpy49s/BISI-Capstone-UI-Design-2026?node-id=0-1&t=77XuAQU0iZNhR8yk-1)**

## Local Development Setup

### 1. Create a Virtual Environment

If you do not already have a virtual environment, create one:

```bash
python -m venv venv
```
### 2. Activate the Virtual Environment

**Windows (PowerShell)**

```powershell
venv\Scripts\Activate
```

### 3. Install Dependencies

Only run this if:

- You are setting up the project for the first time.
- New packages have been added to `requirements.txt`.

```bash
pip install -r requirements.txt
```
#### Code if the venv is not working:
If there are any conflicts in using the requirements.txt I recommend using Python version 3.12 and redoing the first three steps with the following code (make sure the venv that is not working is deactivated):
```bash
py -3.12 -m venv venv312
```
```powershell
venv312\Scripts\Activate
```
```bash
pip install -r requirements.txt
```
### 4. Select the Correct Python Interpreter

In **Visual Studio Code**:

1. Press **Ctrl + Shift + P**.
2. Search for **Python: Select Interpreter**.
3. Select:

```text
venv\Scripts\python.exe
```

## Running the Project

### Run Individual Python Scripts

```bash
python <script_name>.py
```

Example:

```bash
python app.py
```

### Run the Main Application

```bash
python run.py
```

## Building the Executable

Once the application has been tested and is working correctly, build the executable using:

```bash
pyinstaller run.spec
```

If you choose to build the application, make sure the venv is activated. The
Windows build will appear under `dist/Classify & Learn Lab` with the application
name and icon embedded in the executable. Windows shortcuts created from that
executable will use the same icon.
